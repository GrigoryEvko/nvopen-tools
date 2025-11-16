// Function: sub_1791A10
// Address: 0x1791a10
//
__int64 __fastcall sub_1791A10(__int64 a1, __int64 ***a2)
{
  __int64 v3; // r8
  unsigned __int64 v4; // rbx
  int v5; // r12d
  unsigned __int8 v6; // al
  __int64 v7; // r12
  __int64 ****v9; // rdx
  __int64 v10; // r14
  __int64 v11; // r15
  __int64 **v12; // r9
  __int64 v13; // rdi
  __int64 **v14; // r15
  int v15; // r12d
  __int64 ***v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r10
  _QWORD *v21; // rax
  _QWORD *v22; // r15
  __int64 v23; // rdx
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  unsigned __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  unsigned __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdi
  _QWORD *v33; // rax
  __int64 ***v34; // rax
  __int64 *v35; // r15
  _QWORD *v36; // rax
  __int64 v37; // rdx
  unsigned __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rdx
  unsigned __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rdx
  unsigned __int64 v44; // rax
  __int64 v45; // rax
  __int64 ***v46; // rax
  __int64 ***v48; // [rsp+18h] [rbp-68h]
  __int64 **v49; // [rsp+20h] [rbp-60h]
  __int64 v50; // [rsp+20h] [rbp-60h]
  _QWORD *v51; // [rsp+20h] [rbp-60h]
  __int64 **v52; // [rsp+28h] [rbp-58h]
  __int64 v53; // [rsp+28h] [rbp-58h]
  __int64 v54[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v55; // [rsp+40h] [rbp-40h]

  v3 = (__int64)*(a2 - 6);
  v4 = (unsigned __int64)*(a2 - 3);
  v5 = *(unsigned __int8 *)(v3 + 16);
  v6 = *(_BYTE *)(v4 + 16);
  if ( (unsigned __int8)v5 > 0x10u )
  {
    if ( v6 > 0x10u || (unsigned __int8)v5 <= 0x17u || (unsigned int)(v5 - 61) > 1 )
      return 0;
  }
  else
  {
    if ( v6 <= 0x17u )
      return 0;
    v5 = v6;
    v3 = (__int64)*(a2 - 3);
    v4 = (unsigned __int64)*(a2 - 6);
    if ( (unsigned int)v6 - 61 > 1 )
      return 0;
  }
  if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
    v9 = *(__int64 *****)(v3 - 8);
  else
    v9 = (__int64 ****)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF));
  v10 = (__int64)*(a2 - 9);
  v11 = 0;
  v12 = **v9;
  v48 = *v9;
  v13 = (__int64)v12;
  if ( (unsigned __int8)(*(_BYTE *)(v10 + 16) - 75) <= 1u )
    v11 = (__int64)*(a2 - 9);
  if ( *((_BYTE *)v12 + 8) == 16 )
    v13 = *v12[2];
  v49 = **v9;
  v52 = (__int64 **)v3;
  if ( !sub_1642F90(v13, 1) && (!v11 || v49 != **(__int64 ****)(v11 - 48)) )
    return 0;
  v14 = *a2;
  v15 = v5 - 24;
  v16 = (__int64 ***)sub_15A43B0(v4, v49, 0);
  v17 = sub_15A46C0(v15, v16, v14, 0);
  v20 = (__int64)v16;
  if ( v4 != v17 )
  {
    if ( (__int64 ***)v10 != v48 )
      return 0;
    if ( *(a2 - 6) == v52 )
    {
      v34 = (__int64 ***)sub_15A0600((__int64)v49);
      v55 = 257;
      v35 = (__int64 *)sub_15A46C0(v15, v34, v14, 0);
      v36 = sub_1648A60(56, 3u);
      v7 = (__int64)v36;
      if ( v36 )
      {
        v51 = v36 - 9;
        v53 = (__int64)v36;
        sub_15F1EA0((__int64)v36, *v35, 55, (__int64)(v36 - 9), 3, 0);
        if ( *(_QWORD *)(v7 - 72) )
        {
          v37 = *(_QWORD *)(v7 - 64);
          v38 = *(_QWORD *)(v7 - 56) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v38 = v37;
          if ( v37 )
            *(_QWORD *)(v37 + 16) = *(_QWORD *)(v37 + 16) & 3LL | v38;
        }
        *(_QWORD *)(v7 - 72) = v10;
        v39 = *(_QWORD *)(v10 + 8);
        *(_QWORD *)(v7 - 64) = v39;
        if ( v39 )
          *(_QWORD *)(v39 + 16) = (v7 - 64) | *(_QWORD *)(v39 + 16) & 3LL;
        *(_QWORD *)(v7 - 56) = (v10 + 8) | *(_QWORD *)(v7 - 56) & 3LL;
        *(_QWORD *)(v10 + 8) = v51;
        if ( *(_QWORD *)(v7 - 48) )
        {
          v40 = *(_QWORD *)(v7 - 40);
          v41 = *(_QWORD *)(v7 - 32) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v41 = v40;
          if ( v40 )
            *(_QWORD *)(v40 + 16) = *(_QWORD *)(v40 + 16) & 3LL | v41;
        }
        *(_QWORD *)(v7 - 48) = v35;
        v42 = v35[1];
        *(_QWORD *)(v7 - 40) = v42;
        if ( v42 )
          *(_QWORD *)(v42 + 16) = (v7 - 40) | *(_QWORD *)(v42 + 16) & 3LL;
        *(_QWORD *)(v7 - 32) = (unsigned __int64)(v35 + 1) | *(_QWORD *)(v7 - 32) & 3LL;
        v35[1] = v7 - 48;
        if ( *(_QWORD *)(v7 - 24) )
        {
          v43 = *(_QWORD *)(v7 - 16);
          v44 = *(_QWORD *)(v7 - 8) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v44 = v43;
          if ( v43 )
            *(_QWORD *)(v43 + 16) = *(_QWORD *)(v43 + 16) & 3LL | v44;
        }
        *(_QWORD *)(v7 - 24) = v4;
        v45 = *(_QWORD *)(v4 + 8);
        *(_QWORD *)(v7 - 16) = v45;
        if ( v45 )
          *(_QWORD *)(v45 + 16) = (v7 - 16) | *(_QWORD *)(v45 + 16) & 3LL;
        *(_QWORD *)(v7 - 8) = (v4 + 8) | *(_QWORD *)(v7 - 8) & 3LL;
        *(_QWORD *)(v4 + 8) = v7 - 24;
        sub_164B780(v7, v54);
        goto LABEL_40;
      }
    }
    else
    {
      v50 = sub_15A06D0(v14, (__int64)v16, v18, v19);
      v55 = 257;
      v21 = sub_1648A60(56, 3u);
      v7 = (__int64)v21;
      if ( v21 )
      {
        v22 = v21 - 9;
        v53 = (__int64)v21;
        sub_15F1EA0((__int64)v21, *(_QWORD *)v4, 55, (__int64)(v21 - 9), 3, 0);
        if ( *(_QWORD *)(v7 - 72) )
        {
          v23 = *(_QWORD *)(v7 - 64);
          v24 = *(_QWORD *)(v7 - 56) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v24 = v23;
          if ( v23 )
            *(_QWORD *)(v23 + 16) = *(_QWORD *)(v23 + 16) & 3LL | v24;
        }
        *(_QWORD *)(v7 - 72) = v10;
        v25 = *(_QWORD *)(v10 + 8);
        *(_QWORD *)(v7 - 64) = v25;
        if ( v25 )
          *(_QWORD *)(v25 + 16) = (v7 - 64) | *(_QWORD *)(v25 + 16) & 3LL;
        *(_QWORD *)(v7 - 56) = (v10 + 8) | *(_QWORD *)(v7 - 56) & 3LL;
        *(_QWORD *)(v10 + 8) = v22;
        if ( *(_QWORD *)(v7 - 48) )
        {
          v26 = *(_QWORD *)(v7 - 40);
          v27 = *(_QWORD *)(v7 - 32) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v27 = v26;
          if ( v26 )
            *(_QWORD *)(v26 + 16) = *(_QWORD *)(v26 + 16) & 3LL | v27;
        }
        *(_QWORD *)(v7 - 48) = v4;
        v28 = *(_QWORD *)(v4 + 8);
        *(_QWORD *)(v7 - 40) = v28;
        if ( v28 )
          *(_QWORD *)(v28 + 16) = (v7 - 40) | *(_QWORD *)(v28 + 16) & 3LL;
        *(_QWORD *)(v7 - 32) = (v4 + 8) | *(_QWORD *)(v7 - 32) & 3LL;
        *(_QWORD *)(v4 + 8) = v7 - 48;
        if ( *(_QWORD *)(v7 - 24) )
        {
          v29 = *(_QWORD *)(v7 - 16);
          v30 = *(_QWORD *)(v7 - 8) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v30 = v29;
          if ( v29 )
            *(_QWORD *)(v29 + 16) = *(_QWORD *)(v29 + 16) & 3LL | v30;
        }
        *(_QWORD *)(v7 - 24) = v50;
        if ( v50 )
        {
          v31 = *(_QWORD *)(v50 + 8);
          *(_QWORD *)(v7 - 16) = v31;
          if ( v31 )
            *(_QWORD *)(v31 + 16) = (v7 - 16) | *(_QWORD *)(v31 + 16) & 3LL;
          *(_QWORD *)(v7 - 8) = (v50 + 8) | *(_QWORD *)(v7 - 8) & 3LL;
          *(_QWORD *)(v50 + 8) = v7 - 24;
        }
        sub_164B780(v7, v54);
LABEL_40:
        sub_15F4370(v53, (__int64)a2, 0, 0);
        return v7;
      }
    }
    sub_15F4370(0, (__int64)a2, 0, 0);
    return v7;
  }
  if ( *(a2 - 3) == v52 )
  {
    v46 = v48;
    v48 = v16;
    v20 = (__int64)v46;
  }
  v32 = *(_QWORD *)(a1 + 8);
  v55 = 259;
  v54[0] = (__int64)"narrow";
  v33 = sub_1707C10(v32, v10, (__int64)v48, v20, v54, (__int64)a2);
  v55 = 257;
  return sub_15FDBD0(v15, (__int64)v33, (__int64)v14, (__int64)v54, 0);
}
