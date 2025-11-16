// Function: sub_67B7E0
// Address: 0x67b7e0
//
__int64 __fastcall sub_67B7E0(__int64 a1, _QWORD *a2, int *a3, _QWORD *a4, int a5)
{
  char v7; // dl
  __int64 result; // rax
  __int64 v9; // r9
  int v10; // r15d
  int v11; // r15d
  _QWORD *v12; // r14
  char v13; // al
  __int64 *v14; // rdx
  __int64 v15; // rdx
  int v16; // eax
  int v17; // eax
  __int64 v18; // rcx
  char v19; // al
  int v20; // eax
  __int64 v21; // rax
  int v22; // eax
  __int64 v23; // [rsp-60h] [rbp-60h]
  __int64 v24; // [rsp-50h] [rbp-50h] BYREF
  __int64 v25; // [rsp-48h] [rbp-48h] BYREF
  __int64 v26; // [rsp-40h] [rbp-40h] BYREF

  if ( (*(_BYTE *)(a1 + 8) & 0x20) != 0 )
    return 0;
  v7 = *(_BYTE *)(a1 + 4);
  if ( v7 == 9 )
  {
    if ( (*(_BYTE *)(a1 + 6) & 0xA) == 0 )
    {
      v9 = *(_QWORD *)(a1 + 360);
      if ( v9 )
      {
        v10 = -(a5 == 0);
        if ( *(_BYTE *)(v9 + 80) == 9 )
        {
          LOBYTE(v10) = v10 & 0x1F;
          v11 = v10 + 687;
        }
        else
        {
          LOBYTE(v10) = v10 & 0x1B;
          v11 = v10 + 683;
        }
        v12 = (_QWORD *)(a1 + 392);
        v23 = *(_QWORD *)(a1 + 360);
        sub_729E00(*(unsigned int *)(a1 + 392), &v25, &v26, &v24, (char *)&v24 + 4);
        if ( HIDWORD(v24) )
        {
          v13 = *(_BYTE *)(v23 + 80);
          if ( v13 == 9 || (unsigned __int8)(v13 - 10) <= 1u || v13 == 17 )
          {
            v21 = *(_QWORD *)(v23 + 96);
            if ( v21 && *(_DWORD *)(v21 + 92) )
              v12 = (_QWORD *)(v21 + 92);
          }
        }
        *a2 = v23;
        *a3 = v11;
        if ( !a4 )
          return 1;
        goto LABEL_21;
      }
      v15 = *(_QWORD *)(a1 + 368);
      v19 = *(_BYTE *)(v15 + 80);
      if ( v19 == 20 )
      {
        v22 = -(a5 == 0);
        LOBYTE(v22) = v22 & 0x62;
        v17 = v22 + 685;
      }
      else
      {
        if ( (unsigned __int8)(v19 - 21) > 1u && v19 != 19 )
          sub_721090(a1);
        v20 = -(a5 == 0);
        LOBYTE(v20) = v20 & 0x62;
        v17 = v20 + 686;
      }
LABEL_19:
      *a2 = v15;
      *a3 = v17;
      if ( !a4 )
        return 1;
      v12 = (_QWORD *)(a1 + 392);
LABEL_21:
      *a4 = *v12;
      return 1;
    }
    return 0;
  }
  else
  {
    result = 0;
    if ( v7 == 17 )
    {
      v14 = *(__int64 **)(a1 + 216);
      if ( (*((_BYTE *)v14 + 193) & 0x10) != 0 )
      {
        v15 = *v14;
        if ( *(_BYTE *)(v15 + 80) != 10
          || (v18 = *(_QWORD *)(v15 + 88), (*(_BYTE *)(v18 + 194) & 0x40) == 0)
          || !*(_QWORD *)(v18 + 248)
          || (*(_BYTE *)(v18 + 195) & 1) != 0 )
        {
          v16 = -(a5 == 0);
          LOBYTE(v16) = v16 & 0x1B;
          v17 = v16 + 684;
          goto LABEL_19;
        }
      }
    }
  }
  return result;
}
