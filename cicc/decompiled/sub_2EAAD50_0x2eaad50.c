// Function: sub_2EAAD50
// Address: 0x2eaad50
//
__int64 __fastcall sub_2EAAD50(__int64 a1, int a2)
{
  __int64 (*v2)(); // rax
  __int64 v4; // rdi
  __int64 (*v5)(); // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // r8

  v2 = *(__int64 (**)())(*(_QWORD *)a1 + 128LL);
  if ( v2 == sub_2DAC790 )
    BUG();
  v4 = v2();
  v5 = *(__int64 (**)())(*(_QWORD *)v4 + 1288LL);
  if ( v5 == sub_2EAACC0 )
    return 0;
  v7 = ((__int64 (__fastcall *)(__int64))v5)(v4);
  v9 = 16 * v8;
  v10 = v7 + 16 * v8;
  v11 = (16 * v8) >> 6;
  v12 = v9 >> 4;
  if ( v11 > 0 )
  {
    v13 = v7 + (v11 << 6);
    while ( a2 != *(_DWORD *)v7 )
    {
      if ( a2 == *(_DWORD *)(v7 + 16) )
      {
        v7 += 16;
        goto LABEL_11;
      }
      if ( a2 == *(_DWORD *)(v7 + 32) )
      {
        v7 += 32;
        goto LABEL_11;
      }
      if ( a2 == *(_DWORD *)(v7 + 48) )
      {
        v7 += 48;
        goto LABEL_11;
      }
      v7 += 64;
      if ( v13 == v7 )
      {
        v12 = (v10 - v7) >> 4;
        goto LABEL_18;
      }
    }
    goto LABEL_11;
  }
LABEL_18:
  if ( v12 == 2 )
    goto LABEL_19;
  if ( v12 == 3 )
  {
    if ( a2 == *(_DWORD *)v7 )
      goto LABEL_11;
    v7 += 16;
LABEL_19:
    if ( a2 == *(_DWORD *)v7 )
      goto LABEL_11;
    v7 += 16;
    goto LABEL_21;
  }
  if ( v12 != 1 )
    return 0;
LABEL_21:
  v14 = 0;
  if ( a2 != *(_DWORD *)v7 )
    return v14;
LABEL_11:
  if ( v10 == v7 )
    return 0;
  return *(_QWORD *)(v7 + 8);
}
