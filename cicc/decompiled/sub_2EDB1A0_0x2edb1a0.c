// Function: sub_2EDB1A0
// Address: 0x2edb1a0
//
__int64 __fastcall sub_2EDB1A0(__int64 *a1, int a2, __int64 a3, __int64 a4, unsigned __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v9; // rdi
  __int64 v10; // r11
  __int64 (*v13)(); // rax
  char v14; // al
  char v15; // r15
  __int64 result; // rax
  char v17; // al
  __int64 v18; // rdx
  __int64 i; // rax
  __int64 v20; // rcx
  char v21; // si
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rbx
  unsigned int v25; // r15d
  unsigned __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 (*v29)(); // rax
  unsigned __int64 v30; // [rsp+8h] [rbp-68h]
  __int64 v31; // [rsp+18h] [rbp-58h]
  __int64 v32; // [rsp+20h] [rbp-50h]
  unsigned int v34; // [rsp+20h] [rbp-50h]
  __int64 v35; // [rsp+20h] [rbp-50h]
  __int64 v37; // [rsp+28h] [rbp-48h]
  char v38; // [rsp+3Eh] [rbp-32h] BYREF
  char v39[49]; // [rsp+3Fh] [rbp-31h] BYREF

  if ( a4 == a5 )
    return 0;
  v6 = a1[1];
  v9 = a1[5];
  v10 = a3;
  v13 = *(__int64 (**)())(*(_QWORD *)v6 + 192LL);
  if ( v13 != sub_2ED11D0 )
  {
    v17 = ((__int64 (__fastcall *)(__int64, __int64, unsigned __int64, __int64, __int64, __int64))v13)(
            v6,
            a3,
            a5,
            a1[4],
            v9,
            a1[6]);
    v10 = a3;
    if ( !v17 )
      return 0;
    v9 = a1[5];
  }
  v32 = v10;
  sub_2EB3EB0(v9, a5, a4);
  v15 = v14;
  if ( v14 )
  {
    v31 = v32;
    v34 = sub_2E5E7B0(a1[6], a4);
    if ( v34 <= (unsigned int)sub_2E5E7B0(a1[6], a5) )
    {
      v18 = a1[3];
      for ( i = a2 < 0
              ? *(_QWORD *)(*(_QWORD *)(v18 + 56) + 16LL * (a2 & 0x7FFFFFFF) + 8)
              : *(_QWORD *)(*(_QWORD *)(v18 + 304) + 8LL * (unsigned int)a2); ; i = *(_QWORD *)(i + 32) )
      {
        if ( !i )
          return 1;
        if ( (*(_BYTE *)(i + 3) & 0x10) == 0 && (*(_BYTE *)(i + 4) & 8) == 0 )
          break;
      }
      v20 = *(_QWORD *)(i + 16);
      v21 = 0;
      if ( a5 == *(_QWORD *)(v20 + 24) )
        goto LABEL_22;
LABEL_17:
      while ( 1 )
      {
        i = *(_QWORD *)(i + 32);
        if ( !i )
          break;
        while ( (*(_BYTE *)(i + 3) & 0x10) == 0 )
        {
          if ( (*(_BYTE *)(i + 4) & 8) != 0 )
            break;
          v22 = *(_QWORD *)(i + 16);
          if ( v20 == v22 )
            break;
          v20 = *(_QWORD *)(i + 16);
          if ( a5 != *(_QWORD *)(v22 + 24) )
            break;
LABEL_22:
          if ( *(_WORD *)(v20 + 68) == 68 )
            goto LABEL_17;
          i = *(_QWORD *)(i + 32);
          if ( *(_WORD *)(v20 + 68) )
            v21 = v15;
          if ( !i )
            goto LABEL_26;
        }
      }
LABEL_26:
      if ( v21 )
      {
        v38 = 0;
        v23 = sub_2EDA920(a1, v31, a5, &v38, a6);
        if ( v23 )
          return sub_2EDB1A0(a1, (unsigned int)a2, v31, a5, v23, a6);
        v35 = sub_2E5E6D0(a1[6], a4);
        if ( v35 )
        {
          v24 = *(_QWORD *)(v31 + 32);
          v37 = v24 + 40LL * (*(_DWORD *)(v31 + 40) & 0xFFFFFF);
          if ( v37 != v24 )
          {
            while ( 1 )
            {
              if ( !*(_BYTE *)v24 )
              {
                v25 = *(_DWORD *)(v24 + 8);
                if ( v25 )
                {
                  if ( v25 - 1 <= 0x3FFFFFFE )
                  {
                    if ( !(unsigned __int8)sub_2EBF3A0((_QWORD *)a1[3], v25)
                      && (*(_BYTE *)(v24 + 3) & 0x10) == 0
                      && !(unsigned __int8)sub_2EBF3A0((_QWORD *)a1[3], v25) )
                    {
                      v28 = a1[1];
                      v29 = *(__int64 (**)())(*(_QWORD *)v28 + 32LL);
                      if ( v29 == sub_2E4EE60 )
                        return 0;
                      if ( !((unsigned __int8 (__fastcall *)(__int64, __int64))v29)(v28, v24) )
                        return 0;
                    }
                  }
                  else if ( (*(_BYTE *)(v24 + 3) & 0x10) != 0 )
                  {
                    v39[0] = 0;
                    result = sub_2ED2890((__int64)a1, v25, a5, a4, &v38, v39);
                    if ( !(_BYTE)result )
                      return result;
                  }
                  else
                  {
                    v26 = sub_2EBEE10(a1[3], v25);
                    if ( v26 )
                    {
                      v30 = v26;
                      v27 = sub_2E5E6D0(a1[6], *(_QWORD *)(v26 + 24));
                      if ( v35 == v27
                        && (*(_WORD *)(v30 + 68) != 68 && *(_WORD *)(v30 + 68)
                         || *(_DWORD *)(v27 + 16) != 1
                         || *(_QWORD *)(v30 + 24) != **(_QWORD **)(v27 + 8)) )
                      {
                        if ( (unsigned __int8)sub_2ED7930(
                                                (__int64)a1,
                                                1,
                                                *(_QWORD *)(*(_QWORD *)(a1[3] + 56) + 16LL * (v25 & 0x7FFFFFFF))
                                              & 0xFFFFFFFFFFFFFFF8LL,
                                                a5) )
                          return 0;
                      }
                    }
                  }
                }
              }
              v24 += 40;
              if ( v37 == v24 )
                return 1;
            }
          }
          return 1;
        }
        return 0;
      }
    }
  }
  return 1;
}
