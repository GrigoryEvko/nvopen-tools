// Function: sub_1BC6CD0
// Address: 0x1bc6cd0
//
__int64 __fastcall sub_1BC6CD0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  __int64 v5; // rax
  unsigned __int64 v7; // r9
  __int64 v8; // rsi
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 v11; // r10
  __int64 result; // rax
  int v13; // edx
  int v14; // r11d

  v3 = a1 + 576;
  v5 = *(unsigned int *)(a1 + 600);
  if ( (_DWORD)v5 )
  {
    v7 = *(_QWORD *)a2;
    v8 = *(_QWORD *)(a1 + 584);
    v9 = (v5 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v10 = (__int64 *)(v8 + 16LL * v9);
    v11 = *v10;
    if ( v7 == *v10 )
    {
LABEL_3:
      if ( v10 != (__int64 *)(v8 + 16 * v5) )
      {
        *(_QWORD *)sub_1907820(v3, (unsigned __int64 *)a2) = 0;
        result = *(_DWORD *)(*(_QWORD *)a2 + 20LL) & 0xFFFFFFF;
        *(_DWORD *)(a2 + 8) = result;
        return result;
      }
    }
    else
    {
      v13 = 1;
      while ( v11 != -8 )
      {
        v14 = v13 + 1;
        v9 = (v5 - 1) & (v13 + v9);
        v10 = (__int64 *)(v8 + 16LL * v9);
        v11 = *v10;
        if ( v7 == *v10 )
          goto LABEL_3;
        v13 = v14;
      }
    }
  }
  result = sub_1907820(v3, (unsigned __int64 *)a2);
  *(_QWORD *)result = a3;
  return result;
}
