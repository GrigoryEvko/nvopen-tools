// Function: sub_2F50FE0
// Address: 0x2f50fe0
//
__int64 __fastcall sub_2F50FE0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // r14d
  __int64 v7; // rdx
  __int64 v9; // rdi
  unsigned __int64 v10; // r15
  __int64 v11; // rbx
  __int64 v12; // rdi
  __int64 v13; // rbx
  unsigned int v14; // [rsp+Ch] [rbp-44h]
  __int64 v15; // [rsp+18h] [rbp-38h]

  v4 = *(_DWORD *)(a3 + 64);
  if ( (unsigned int)a4 <= 0xFE )
  {
    v7 = a1[6];
    v9 = a1[8];
    v10 = *(_QWORD *)(*(_QWORD *)(v7 + 56) + 16LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
    v11 = *(_QWORD *)v9 + 24LL * *(unsigned __int16 *)(*(_QWORD *)v10 + 24LL);
    if ( *(_DWORD *)(v9 + 8) == *(_DWORD *)v11 )
    {
      if ( *(unsigned __int8 *)(v11 + 9) < (unsigned int)a4 )
        goto LABEL_6;
    }
    else
    {
      v14 = a4;
      sub_2F60630(v9, v10, 3LL * *(unsigned __int16 *)(*(_QWORD *)v10 + 24LL), a4);
      a4 = v14;
      if ( *(unsigned __int8 *)(v11 + 9) < v14 )
      {
LABEL_6:
        if ( *(unsigned __int8 *)(a1[9] + *(unsigned __int16 *)(*(_QWORD *)(a3 + 56) + 2LL * *(_QWORD *)(a3 + 64) - 2)) >= (unsigned int)a4 )
        {
          v12 = a1[8];
          v13 = *(_QWORD *)v12 + 24LL * *(unsigned __int16 *)(*(_QWORD *)v10 + 24LL);
          if ( *(_DWORD *)(v12 + 8) != *(_DWORD *)v13 )
            sub_2F60630(v12, v10, 3LL * *(unsigned __int16 *)(*(_QWORD *)v10 + 24LL), a4);
          v4 = *(unsigned __int16 *)(v13 + 10);
        }
        goto LABEL_2;
      }
    }
    BYTE4(v15) = 0;
    return v15;
  }
LABEL_2:
  LODWORD(v15) = v4;
  BYTE4(v15) = 1;
  return v15;
}
