// Function: sub_399C6D0
// Address: 0x399c6d0
//
__int64 __fastcall sub_399C6D0(__int64 a1)
{
  char *v2; // rbx
  __int64 v3; // rax
  _QWORD *v4; // r14
  int v5; // eax
  _QWORD *v6; // rax
  __int64 v7; // r13
  _QWORD *v8; // r15
  _QWORD *v9; // rax
  unsigned int v10; // r11d
  __int64 v11; // rdi
  unsigned __int64 v12; // rbx
  int v13; // eax
  int v14; // ebx
  __int64 v15; // rax
  __int64 result; // rax
  const void *v17; // rsi
  char *v18; // rdi
  size_t v19; // rdx
  unsigned int v20; // [rsp+8h] [rbp-138h]
  __int64 v21; // [rsp+10h] [rbp-130h]
  __int64 v22; // [rsp+18h] [rbp-128h]
  __int64 v23; // [rsp+20h] [rbp-120h]
  __int64 v24; // [rsp+20h] [rbp-120h]
  unsigned int v25; // [rsp+20h] [rbp-120h]
  _QWORD *v26; // [rsp+28h] [rbp-118h]
  char *v27; // [rsp+38h] [rbp-108h]
  unsigned __int64 v28; // [rsp+40h] [rbp-100h] BYREF
  int v29; // [rsp+48h] [rbp-F8h]
  const char *v30; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v31; // [rsp+58h] [rbp-E8h]
  _BYTE dest[136]; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v33; // [rsp+E8h] [rbp-58h]
  __int64 v34; // [rsp+F0h] [rbp-50h]
  unsigned __int64 v35; // [rsp+F8h] [rbp-48h]
  __int64 v36; // [rsp+100h] [rbp-40h]
  int v37; // [rsp+108h] [rbp-38h]

  v26 = (_QWORD *)sub_396DD80(*(_QWORD *)(a1 + 8));
  sub_399C630(a1);
  sub_3989D50(a1);
  v2 = *(char **)(a1 + 552);
  v22 = 0;
  v21 = 0;
  v27 = *(char **)(a1 + 560);
  if ( (unsigned __int64)(v27 - v2) > 0x10 )
  {
    v3 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 232LL);
    v22 = *(_QWORD *)(v3 + 880);
    v21 = *(_QWORD *)(v3 + 888);
  }
  for ( ; v27 != v2; v2 += 16 )
  {
    v7 = *((_QWORD *)v2 + 1);
    if ( *(_DWORD *)(*(_QWORD *)(v7 + 80) + 36LL) != 3 )
    {
      v4 = (_QWORD *)*((_QWORD *)v2 + 1);
      sub_39A3C10(v4);
      v8 = *(_QWORD **)(v7 + 616);
      if ( !*(_BYTE *)(a1 + 4513) )
        goto LABEL_5;
      v23 = *(_QWORD *)(a1 + 8);
      sub_16C1840(&v30);
      v34 = 0;
      v35 = 0;
      v33 = v23;
      v36 = 0;
      v37 = 0;
      v24 = sub_39C70A0(&v30, v22, v21, v7 + 8);
      j___libc_free_0(v35);
      if ( (unsigned __int16)sub_398C0A0(a1) <= 4u )
      {
        LODWORD(v30) = 65543;
        sub_39A3560(v7, v7 + 16, 8497, &v30, v24);
        LODWORD(v30) = 65543;
        sub_39A3560(v8, v8 + 2, 8497, &v30, v24);
      }
      else
      {
        *(_QWORD *)(v7 + 928) = v24;
        v8[116] = v24;
      }
      if ( *(_DWORD *)(a1 + 5528) )
        sub_39A3E10(v8, v8 + 1, 8499, *(_QWORD *)(v26[36] + 8LL), *(_QWORD *)(v26[36] + 8LL));
      if ( (unsigned __int16)sub_398C0A0(a1) <= 4u )
      {
        v9 = (_QWORD *)v8[77];
        v4 = v8;
        if ( !v9 )
          v9 = v8;
        if ( *((_DWORD *)v9 + 186) )
          sub_39A3E10(v8, v8 + 1, 8498, *(_QWORD *)(v26[20] + 8LL), *(_QWORD *)(v26[20] + 8LL));
      }
      else
      {
LABEL_5:
        if ( v8 )
          v4 = v8;
        else
          v8 = (_QWORD *)v7;
      }
      v5 = *(_DWORD *)(v7 + 816);
      if ( v5 )
      {
        if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 232LL) + 504LL) - 34) > 1 )
        {
          if ( v5 != 1 && *(_BYTE *)(a1 + 4501) )
          {
            LODWORD(v30) = 65537;
            sub_39A3560(v4, v8 + 2, 17, &v30, 0);
          }
          else
          {
            v8[107] = **(_QWORD **)(v7 + 808);
          }
          v30 = dest;
          v31 = 0x200000000LL;
          v10 = *(_DWORD *)(v7 + 816);
          if ( v10 && &v30 != (const char **)(v7 + 808) )
          {
            v17 = (const void *)(v7 + 824);
            if ( *(_QWORD *)(v7 + 808) == v7 + 824 )
            {
              v18 = dest;
              v19 = 16LL * v10;
              if ( v10 <= 2
                || (v20 = *(_DWORD *)(v7 + 816),
                    sub_16CD150((__int64)&v30, dest, v10, 16, (int)dest, v10),
                    v18 = (char *)v30,
                    v17 = *(const void **)(v7 + 808),
                    v19 = 16LL * *(unsigned int *)(v7 + 816),
                    v10 = v20,
                    v19) )
              {
                v25 = v10;
                memcpy(v18, v17, v19);
                v10 = v25;
              }
              LODWORD(v31) = v10;
              *(_DWORD *)(v7 + 816) = 0;
            }
            else
            {
              v30 = *(const char **)(v7 + 808);
              LODWORD(v31) = v10;
              HIDWORD(v31) = *(_DWORD *)(v7 + 820);
              *(_QWORD *)(v7 + 808) = v17;
              *(_QWORD *)(v7 + 816) = 0;
            }
          }
          sub_39CB220(v8, v8 + 1, &v30);
          if ( v30 != dest )
            _libc_free((unsigned __int64)v30);
        }
        else
        {
          LODWORD(v30) = 65537;
          sub_39A3560(v4, v8 + 2, 17, &v30, 0);
        }
      }
      if ( (unsigned __int16)sub_398C0A0(a1) > 4u && !*(_BYTE *)(a1 + 4513) )
      {
        v6 = (_QWORD *)v8[77];
        if ( !v6 )
          v6 = v8;
        if ( *((_DWORD *)v6 + 186) )
          sub_39A3EE0(v4);
      }
      if ( *(_QWORD *)(*(_QWORD *)v2 + 8 * (8LL - *(unsigned int *)(*(_QWORD *)v2 + 8LL))) )
        sub_39A3E10(v4, v8 + 1, 67, v8[79], *(_QWORD *)(v26[21] + 8LL));
      sub_39A26B0(v4);
    }
  }
  v11 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 1688LL);
  dest[1] = 1;
  v30 = "llvm.dbg.cu";
  dest[0] = 3;
  v12 = sub_1632310(v11, (__int64)&v30);
  v13 = 0;
  if ( v12 )
    v13 = sub_161F520(v12);
  LODWORD(v31) = v13;
  v30 = (const char *)v12;
  sub_1632FD0((__int64)&v30);
  v28 = v12;
  v29 = 0;
  sub_1632FD0((__int64)&v28);
  v14 = v31;
  LODWORD(v31) = v29;
  v30 = (const char *)v28;
  if ( v29 != v14 )
  {
    do
    {
      v15 = sub_1632FB0((__int64)&v30);
      if ( *(_QWORD *)(v15 + 40) )
        sub_3999410(a1, v15);
      LODWORD(v31) = v31 + 1;
      sub_1632FD0((__int64)&v30);
    }
    while ( (_DWORD)v31 != v14 );
  }
  result = sub_39A02E0(a1 + 4040);
  if ( *(_BYTE *)(a1 + 4513) )
    return sub_39A02E0(a1 + 4520);
  return result;
}
