// Function: sub_15D4830
// Address: 0x15d4830
//
__int64 __fastcall sub_15D4830(_QWORD *a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r12
  unsigned __int64 v7; // rdi
  __int64 v9; // rax
  __int64 v10; // r12
  unsigned int v11; // esi
  char v12; // dl
  __int64 v13; // [rsp+18h] [rbp-2B8h]
  __int64 v14; // [rsp+28h] [rbp-2A8h]
  unsigned int v15; // [rsp+30h] [rbp-2A0h]
  __int64 v17; // [rsp+38h] [rbp-298h] BYREF
  __int64 v18; // [rsp+40h] [rbp-290h] BYREF
  __int64 v19; // [rsp+48h] [rbp-288h] BYREF
  __int64 v20; // [rsp+50h] [rbp-280h] BYREF
  __int64 v21; // [rsp+58h] [rbp-278h] BYREF
  _BYTE *v22; // [rsp+60h] [rbp-270h] BYREF
  __int64 v23; // [rsp+68h] [rbp-268h]
  _BYTE v24[256]; // [rsp+70h] [rbp-260h] BYREF
  __int64 v25; // [rsp+170h] [rbp-160h] BYREF
  _BYTE *v26; // [rsp+178h] [rbp-158h]
  _BYTE *v27; // [rsp+180h] [rbp-150h]
  __int64 v28; // [rsp+188h] [rbp-148h]
  int v29; // [rsp+190h] [rbp-140h]
  _BYTE v30[312]; // [rsp+198h] [rbp-138h] BYREF

  v4 = (__int64)(a1 + 3);
  v17 = a2;
  v5 = sub_15D4720((__int64)(a1 + 3), &v17);
  v6 = v17;
  v13 = v5;
  if ( *(_DWORD *)(v5 + 8) < a3 )
    return v6;
  v25 = 0;
  v22 = v24;
  v23 = 0x2000000000LL;
  v26 = v30;
  v27 = v30;
  v28 = 32;
  v29 = 0;
  if ( *(_DWORD *)(v5 + 12) >= a3 )
  {
    sub_15CDD90((__int64)&v22, &v17);
    v9 = (unsigned int)v23;
    if ( !(_DWORD)v23 )
    {
LABEL_16:
      v6 = *(_QWORD *)(v13 + 24);
      if ( v27 != v26 )
        _libc_free((unsigned __int64)v27);
      v7 = (unsigned __int64)v22;
      if ( v22 != v24 )
        goto LABEL_4;
      return v6;
    }
    while ( 1 )
    {
      v18 = *(_QWORD *)&v22[8 * v9 - 8];
      v10 = sub_15D4720(v4, &v18);
      v19 = *(_QWORD *)(*a1 + 8LL * *(unsigned int *)(v10 + 12));
      sub_1412190((__int64)&v25, v19);
      v11 = *(_DWORD *)(v10 + 12);
      if ( v12 )
        break;
      v9 = (unsigned int)(v23 - 1);
      LODWORD(v23) = v23 - 1;
      if ( a3 > v11 )
      {
LABEL_10:
        if ( !(_DWORD)v9 )
          goto LABEL_16;
      }
      else
      {
        v14 = sub_15D4720(v4, &v19);
        v20 = *(_QWORD *)(v14 + 24);
        v21 = *(_QWORD *)(v10 + 24);
        v15 = *(_DWORD *)(sub_15D4720(v4, &v20) + 16);
        if ( v15 < *(_DWORD *)(sub_15D4720(v4, &v21) + 16) )
          *(_QWORD *)(v10 + 24) = v20;
        *(_DWORD *)(v10 + 12) = *(_DWORD *)(v14 + 12);
        v9 = (unsigned int)v23;
        if ( !(_DWORD)v23 )
          goto LABEL_16;
      }
    }
    if ( a3 <= v11 )
    {
      sub_15CDD90((__int64)&v22, &v19);
      v9 = (unsigned int)v23;
    }
    else
    {
      v9 = (unsigned int)(v23 - 1);
      LODWORD(v23) = v23 - 1;
    }
    goto LABEL_10;
  }
  v7 = (unsigned __int64)v22;
  v6 = *(_QWORD *)(v5 + 24);
  if ( v22 != v24 )
LABEL_4:
    _libc_free(v7);
  return v6;
}
