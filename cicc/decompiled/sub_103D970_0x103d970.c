// Function: sub_103D970
// Address: 0x103d970
//
__int64 __fastcall sub_103D970(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 result; // rax
  int v8; // edi
  unsigned int v9; // edx
  __int64 v10; // r8
  __int64 v11; // r14
  __int64 v12; // rax
  _WORD *v13; // rdx
  __int64 v14; // r13
  __int64 v15; // rsi
  unsigned int v16; // r9d

  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(_QWORD *)(v3 + 40);
  result = *(unsigned int *)(v3 + 56);
  if ( (_DWORD)result )
  {
    v8 = result - 1;
    v9 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    result = v4 + 16LL * v9;
    v10 = *(_QWORD *)result;
    if ( a2 == *(_QWORD *)result )
    {
LABEL_3:
      v11 = *(_QWORD *)(result + 8);
      if ( v11 )
      {
        v12 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 16) + 16LL))(
                *(_QWORD *)(a1 + 16),
                *(_QWORD *)(result + 8),
                a1 + 24);
        v13 = *(_WORD **)(a3 + 32);
        v14 = v12;
        if ( *(_QWORD *)(a3 + 24) - (_QWORD)v13 <= 1u )
        {
          v15 = sub_CB6200(a3, (unsigned __int8 *)"; ", 2u);
        }
        else
        {
          v15 = a3;
          *v13 = 8251;
          *(_QWORD *)(a3 + 32) += 2LL;
        }
        sub_103D830(v11, v15);
        if ( v14 )
        {
          sub_904010(a3, " - clobbered by ");
          if ( v14 == *(_QWORD *)(*(_QWORD *)(a1 + 8) + 128LL) )
            sub_904010(a3, "liveOnEntry");
          else
            sub_103D830(v14, a3);
        }
        result = *(_QWORD *)(a3 + 32);
        if ( *(_QWORD *)(a3 + 24) == result )
        {
          return sub_CB6200(a3, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *(_BYTE *)result = 10;
          ++*(_QWORD *)(a3 + 32);
        }
      }
    }
    else
    {
      result = 1;
      while ( v10 != -4096 )
      {
        v16 = result + 1;
        v9 = v8 & (result + v9);
        result = v4 + 16LL * v9;
        v10 = *(_QWORD *)result;
        if ( a2 == *(_QWORD *)result )
          goto LABEL_3;
        result = v16;
      }
    }
  }
  return result;
}
