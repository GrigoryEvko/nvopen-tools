// Function: sub_39C8280
// Address: 0x39c8280
//
__int64 __fastcall sub_39C8280(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // rsi
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 v11; // r8
  int v13; // edx
  int v14; // r10d

  *a4 = a2;
  if ( !sub_39C7370(a1) || (unsigned __int8)sub_3989C80(*(_QWORD *)(a1 + 200)) )
    v5 = *(_QWORD *)(a1 + 208) + 328LL;
  else
    v5 = a1 + 896;
  v6 = *(unsigned int *)(v5 + 24);
  v7 = 0;
  if ( (_DWORD)v6 )
  {
    v8 = *(_QWORD *)(v5 + 8);
    v9 = (v6 - 1) & (((unsigned int)*a4 >> 9) ^ ((unsigned int)*a4 >> 4));
    v10 = (__int64 *)(v8 + 16LL * v9);
    v11 = *v10;
    if ( *a4 == *v10 )
    {
LABEL_5:
      if ( v10 != (__int64 *)(v8 + 16 * v6) )
        return v10[1];
    }
    else
    {
      v13 = 1;
      while ( v11 != -8 )
      {
        v14 = v13 + 1;
        v9 = (v6 - 1) & (v13 + v9);
        v10 = (__int64 *)(v8 + 16LL * v9);
        v11 = *v10;
        if ( *a4 == *v10 )
          goto LABEL_5;
        v13 = v14;
      }
    }
    return 0;
  }
  return v7;
}
