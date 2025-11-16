// Function: sub_3171820
// Address: 0x3171820
//
_BYTE **__fastcall sub_3171820(__int64 a1, __int64 a2, __int64 *a3)
{
  _BYTE **result; // rax
  _BYTE *v5; // rsi
  char **v6; // r13
  __int64 v7; // rsi
  char v8; // al
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  char *v14; // rbx
  unsigned __int8 v15; // dl
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // rdx
  _BYTE *v21; // r8
  _BYTE *v22; // r13
  __int64 v23; // rsi
  char v24; // al
  __int64 v25; // r9
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rbx
  char *v31; // rdx
  unsigned __int8 v32; // cl
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // r8
  __int64 v37; // rbx
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // rax
  __int64 v41; // rdx
  _BYTE **v42; // [rsp+10h] [rbp-190h]
  _BYTE **v43; // [rsp+28h] [rbp-178h]
  char *v44; // [rsp+30h] [rbp-170h]
  __int64 v45; // [rsp+30h] [rbp-170h]
  __int64 v46; // [rsp+30h] [rbp-170h]
  char **v47; // [rsp+38h] [rbp-168h]
  _BYTE *v48; // [rsp+38h] [rbp-168h]
  _BYTE *v49; // [rsp+48h] [rbp-158h] BYREF
  char **v50; // [rsp+50h] [rbp-150h] BYREF
  __int64 v51; // [rsp+58h] [rbp-148h]
  _BYTE v52[128]; // [rsp+60h] [rbp-140h] BYREF
  _BYTE *v53; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v54; // [rsp+E8h] [rbp-B8h]
  _BYTE v55[176]; // [rsp+F0h] [rbp-B0h] BYREF

  result = *(_BYTE ***)(a1 + 144);
  v42 = &result[5 * *(unsigned int *)(a1 + 152)];
  if ( result == v42 )
    return result;
  v43 = *(_BYTE ***)(a1 + 144);
  do
  {
    v5 = *v43;
    v53 = v55;
    v50 = (char **)v52;
    v51 = 0x1000000000LL;
    v54 = 0x1000000000LL;
    v49 = v5;
    sub_AE7A40((__int64)&v50, v5, (__int64)&v53);
    v6 = v50;
    v47 = &v50[(unsigned int)v51];
    if ( v47 != v50 )
    {
      do
      {
        while ( 1 )
        {
          v14 = *v6;
          v15 = *v49;
          if ( *v49 != 22 )
            break;
          v7 = *(_QWORD *)(*((_QWORD *)v49 + 3) + 80LL);
          if ( v7 )
            v7 -= 24;
          v8 = *v14;
          if ( *v14 != 84 )
            goto LABEL_16;
LABEL_8:
          if ( (*((_DWORD *)v14 + 1) & 0x7FFFFFFu) > 1 )
            goto LABEL_11;
          v9 = *((_QWORD *)v14 + 5);
LABEL_10:
          if ( !sub_24F96E0(a3, v7, v9) )
            goto LABEL_11;
LABEL_25:
          v18 = sub_31711D0(a1, (__int64 *)&v49, v10, v11, v12, v13);
          v20 = *(unsigned int *)(v18 + 8);
          if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(v18 + 12) )
          {
            v45 = v18;
            sub_C8D5F0(v18, (const void *)(v18 + 16), v20 + 1, 8u, v19, v20 + 1);
            v18 = v45;
            v20 = *(unsigned int *)(v45 + 8);
          }
          ++v6;
          *(_QWORD *)(*(_QWORD *)v18 + 8 * v20) = v14;
          ++*(_DWORD *)(v18 + 8);
          if ( v47 == v6 )
            goto LABEL_28;
        }
        if ( v15 <= 0x1Cu )
LABEL_78:
          BUG();
        v7 = *((_QWORD *)v49 + 5);
        if ( v15 == 85 )
        {
          v41 = *((_QWORD *)v49 - 4);
          if ( v41 )
          {
            if ( !*(_BYTE *)v41
              && *(_QWORD *)(v41 + 24) == *((_QWORD *)v49 + 10)
              && (*(_BYTE *)(v41 + 33) & 0x20) != 0
              && (unsigned int)(*(_DWORD *)(v41 + 36) - 60) <= 2 )
            {
              v7 = sub_AA56F0(*((_QWORD *)v49 + 5));
            }
          }
        }
        v8 = *v14;
        if ( *v14 == 84 )
          goto LABEL_8;
LABEL_16:
        v9 = *((_QWORD *)v14 + 5);
        if ( v8 != 85 )
          goto LABEL_10;
        v16 = *((_QWORD *)v14 - 4);
        if ( !v16 )
          goto LABEL_10;
        if ( (*(_BYTE *)v16
           || *(_QWORD *)(v16 + 24) != *((_QWORD *)v14 + 10)
           || (*(_BYTE *)(v16 + 33) & 0x20) == 0
           || *(_DWORD *)(v16 + 36) != 62)
          && (*(_BYTE *)v16
           || *(_QWORD *)(v16 + 24) != *((_QWORD *)v14 + 10)
           || (*(_BYTE *)(v16 + 33) & 0x20) == 0
           || *(_DWORD *)(v16 + 36) != 61) )
        {
          goto LABEL_10;
        }
        v17 = sub_AA54C0(*((_QWORD *)v14 + 5));
        if ( sub_24F96E0(a3, v7, v17) )
          goto LABEL_25;
LABEL_11:
        ++v6;
      }
      while ( v47 != v6 );
    }
LABEL_28:
    v21 = v53;
    v22 = v53;
    v48 = &v53[8 * (unsigned int)v54];
    if ( v48 != v53 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v30 = *(_QWORD *)v22;
          v31 = **(char ***)(*(_QWORD *)v22 + 16LL);
          v32 = *v49;
          if ( *v49 != 22 )
            break;
          v23 = *(_QWORD *)(*((_QWORD *)v49 + 3) + 80LL);
          if ( v23 )
            v23 -= 24;
          v24 = *v31;
          if ( *v31 != 84 )
            goto LABEL_41;
LABEL_33:
          if ( (*((_DWORD *)v31 + 1) & 0x7FFFFFFu) > 1 )
            goto LABEL_36;
          v25 = *((_QWORD *)v31 + 5);
LABEL_35:
          if ( !sub_24F96E0(a3, v23, v25) )
            goto LABEL_36;
LABEL_50:
          v35 = sub_31711D0(a1, (__int64 *)&v49, v26, v27, v28, v29);
          v37 = **(_QWORD **)(v30 + 16);
          v38 = *(unsigned int *)(v35 + 8);
          if ( v38 + 1 > (unsigned __int64)*(unsigned int *)(v35 + 12) )
          {
            v46 = v35;
            sub_C8D5F0(v35, (const void *)(v35 + 16), v38 + 1, 8u, v36, v38 + 1);
            v35 = v46;
            v38 = *(unsigned int *)(v46 + 8);
          }
          v22 += 8;
          *(_QWORD *)(*(_QWORD *)v35 + 8 * v38) = v37;
          ++*(_DWORD *)(v35 + 8);
          if ( v48 == v22 )
          {
LABEL_53:
            v21 = v53;
            goto LABEL_54;
          }
        }
        if ( v32 <= 0x1Cu )
          goto LABEL_78;
        v23 = *((_QWORD *)v49 + 5);
        if ( v32 == 85 )
        {
          v39 = *((_QWORD *)v49 - 4);
          if ( v39 )
          {
            if ( !*(_BYTE *)v39
              && *(_QWORD *)(v39 + 24) == *((_QWORD *)v49 + 10)
              && (*(_BYTE *)(v39 + 33) & 0x20) != 0
              && (unsigned int)(*(_DWORD *)(v39 + 36) - 60) <= 2 )
            {
              v44 = **(char ***)(*(_QWORD *)v22 + 16LL);
              v40 = sub_AA56F0(*((_QWORD *)v49 + 5));
              v31 = v44;
              v23 = v40;
            }
          }
        }
        v24 = *v31;
        if ( *v31 == 84 )
          goto LABEL_33;
LABEL_41:
        v25 = *((_QWORD *)v31 + 5);
        if ( v24 != 85 )
          goto LABEL_35;
        v33 = *((_QWORD *)v31 - 4);
        if ( !v33 )
          goto LABEL_35;
        if ( (*(_BYTE *)v33
           || *(_QWORD *)(v33 + 24) != *((_QWORD *)v31 + 10)
           || (*(_BYTE *)(v33 + 33) & 0x20) == 0
           || *(_DWORD *)(v33 + 36) != 62)
          && (*(_BYTE *)v33
           || *(_QWORD *)(v33 + 24) != *((_QWORD *)v31 + 10)
           || (*(_BYTE *)(v33 + 33) & 0x20) == 0
           || *(_DWORD *)(v33 + 36) != 61) )
        {
          goto LABEL_35;
        }
        v34 = sub_AA54C0(*((_QWORD *)v31 + 5));
        if ( sub_24F96E0(a3, v23, v34) )
          goto LABEL_50;
LABEL_36:
        v22 += 8;
        if ( v48 == v22 )
          goto LABEL_53;
      }
    }
LABEL_54:
    if ( v21 != v55 )
      _libc_free((unsigned __int64)v21);
    if ( v50 != (char **)v52 )
      _libc_free((unsigned __int64)v50);
    v43 += 5;
    result = v43;
  }
  while ( v42 != v43 );
  return result;
}
