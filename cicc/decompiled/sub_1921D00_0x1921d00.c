// Function: sub_1921D00
// Address: 0x1921d00
//
_BOOL8 __fastcall sub_1921D00(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 *v7; // rbx
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 v10; // r12
  __int64 v11; // r15
  __int64 *v12; // rbx
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // r14
  __int64 *v16; // rbx
  __int64 v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // r13
  __int64 *v20; // r14
  __int64 *v21; // rbx
  __int64 v22; // rax
  __int64 *v23; // rdx
  __int64 *v24; // r14
  __int64 *v25; // rbx
  __int64 v26; // rax
  __int64 *v27; // rdx
  __int64 v28; // r14
  __int64 *v29; // rbx
  __int64 v30; // rax
  __int64 *v31; // r12
  __int64 v32; // r13
  __int64 *v33; // rbx
  __int64 v34; // rax
  __int64 *v35; // rcx
  __int64 v36; // rbx
  __int64 *v37; // r12
  __int64 *v38; // r14
  __int64 v39; // rax
  __int64 *v40; // r13
  __int64 *v41; // rcx
  __int64 v42; // r12
  __int64 *v43; // rbx
  __int64 v44; // r14
  __int64 *v46; // [rsp+8h] [rbp-B8h]
  __int64 *v47; // [rsp+10h] [rbp-B0h]
  __int64 *v48; // [rsp+18h] [rbp-A8h]
  __int64 *v49; // [rsp+20h] [rbp-A0h]
  __int64 *v50; // [rsp+28h] [rbp-98h]
  __int64 *v51; // [rsp+30h] [rbp-90h]
  __int64 *v52; // [rsp+38h] [rbp-88h]
  __int64 *v53; // [rsp+40h] [rbp-80h]
  __int64 *v54; // [rsp+48h] [rbp-78h]
  __int64 *v55; // [rsp+50h] [rbp-70h]
  bool v56; // [rsp+5Fh] [rbp-61h]
  __int64 *v57; // [rsp+60h] [rbp-60h]
  __int64 *v58; // [rsp+68h] [rbp-58h]
  __int64 *v59; // [rsp+70h] [rbp-50h]
  __int64 *v60; // [rsp+78h] [rbp-48h]
  __int64 *v61; // [rsp+80h] [rbp-40h]
  __int64 *v62; // [rsp+88h] [rbp-38h]

  v5 = 24LL * (*((_DWORD *)a2 + 5) & 0xFFFFFFF);
  if ( (*((_BYTE *)a2 + 23) & 0x40) != 0 )
  {
    v6 = (__int64 *)*(a2 - 1);
    v59 = &v6[(unsigned __int64)v5 / 8];
  }
  else
  {
    v59 = a2;
    v6 = &a2[v5 / 0xFFFFFFFFFFFFFFF8LL];
  }
  v62 = v6;
  if ( v6 == v59 )
  {
    return 1;
  }
  else
  {
    while ( 1 )
    {
      v7 = (__int64 *)*v62;
      if ( *(_BYTE *)(*v62 + 16) > 0x17u )
      {
        v56 = sub_15CC8F0(*(_QWORD *)(a1 + 216), v7[5], a3);
        if ( !v56 )
        {
          if ( *((_BYTE *)v7 + 16) != 56 )
            return v56;
          v8 = 24LL * (*((_DWORD *)v7 + 5) & 0xFFFFFFF);
          if ( (*((_BYTE *)v7 + 23) & 0x40) != 0 )
          {
            v9 = (__int64 *)*(v7 - 1);
            v57 = &v9[(unsigned __int64)v8 / 8];
          }
          else
          {
            v57 = v7;
            v9 = &v7[v8 / 0xFFFFFFFFFFFFFFF8LL];
          }
          v61 = v9;
          if ( v9 != v57 )
            break;
        }
      }
LABEL_80:
      v62 += 3;
      if ( v59 == v62 )
        return 1;
    }
    v10 = a1;
    v11 = a3;
    while ( 1 )
    {
      v12 = (__int64 *)*v61;
      if ( *(_BYTE *)(*v61 + 16) > 0x17u && !sub_15CC8F0(*(_QWORD *)(v10 + 216), v12[5], v11) )
      {
        if ( *((_BYTE *)v12 + 16) != 56 )
          return v56;
        v13 = 24LL * (*((_DWORD *)v12 + 5) & 0xFFFFFFF);
        if ( (*((_BYTE *)v12 + 23) & 0x40) != 0 )
        {
          v14 = (__int64 *)*(v12 - 1);
          v55 = &v14[(unsigned __int64)v13 / 8];
        }
        else
        {
          v55 = v12;
          v14 = &v12[v13 / 0xFFFFFFFFFFFFFFF8LL];
        }
        v60 = v14;
        if ( v55 != v14 )
          break;
      }
LABEL_78:
      v61 += 3;
      if ( v57 == v61 )
      {
        a3 = v11;
        a1 = v10;
        goto LABEL_80;
      }
    }
    v15 = v10;
    while ( 1 )
    {
      v16 = (__int64 *)*v60;
      if ( *(_BYTE *)(*v60 + 16) > 0x17u && !sub_15CC8F0(*(_QWORD *)(v15 + 216), v16[5], v11) )
      {
        if ( *((_BYTE *)v16 + 16) != 56 )
          return v56;
        v17 = 24LL * (*((_DWORD *)v16 + 5) & 0xFFFFFFF);
        if ( (*((_BYTE *)v16 + 23) & 0x40) != 0 )
        {
          v18 = (__int64 *)*(v16 - 1);
          v54 = &v18[(unsigned __int64)v17 / 8];
        }
        else
        {
          v54 = v16;
          v18 = &v16[v17 / 0xFFFFFFFFFFFFFFF8LL];
        }
        if ( v54 != v18 )
          break;
      }
LABEL_76:
      v60 += 3;
      if ( v55 == v60 )
      {
        v10 = v15;
        goto LABEL_78;
      }
    }
    v19 = v15;
    v20 = v18;
    while ( 1 )
    {
      v21 = (__int64 *)*v20;
      if ( *(_BYTE *)(*v20 + 16) > 0x17u && !sub_15CC8F0(*(_QWORD *)(v19 + 216), v21[5], v11) )
      {
        if ( *((_BYTE *)v21 + 16) != 56 )
          return v56;
        v22 = 24LL * (*((_DWORD *)v21 + 5) & 0xFFFFFFF);
        if ( (*((_BYTE *)v21 + 23) & 0x40) != 0 )
        {
          v23 = (__int64 *)*(v21 - 1);
          v53 = &v23[(unsigned __int64)v22 / 8];
        }
        else
        {
          v53 = v21;
          v23 = &v21[v22 / 0xFFFFFFFFFFFFFFF8LL];
        }
        if ( v23 != v53 )
          break;
      }
LABEL_74:
      v20 += 3;
      if ( v54 == v20 )
      {
        v15 = v19;
        goto LABEL_76;
      }
    }
    v49 = v20;
    v24 = v23;
    while ( 1 )
    {
      v25 = (__int64 *)*v24;
      if ( *(_BYTE *)(*v24 + 16) > 0x17u && !sub_15CC8F0(*(_QWORD *)(v19 + 216), v25[5], v11) )
      {
        if ( *((_BYTE *)v25 + 16) != 56 )
          return v56;
        v26 = 24LL * (*((_DWORD *)v25 + 5) & 0xFFFFFFF);
        if ( (*((_BYTE *)v25 + 23) & 0x40) != 0 )
        {
          v27 = (__int64 *)*(v25 - 1);
          v52 = &v27[(unsigned __int64)v26 / 8];
        }
        else
        {
          v52 = v25;
          v27 = &v25[v26 / 0xFFFFFFFFFFFFFFF8LL];
        }
        v58 = v27;
        if ( v27 != v52 )
          break;
      }
LABEL_72:
      v24 += 3;
      if ( v53 == v24 )
      {
        v20 = v49;
        goto LABEL_74;
      }
    }
    v48 = v24;
    v28 = v19;
    while ( 1 )
    {
      v29 = (__int64 *)*v58;
      if ( *(_BYTE *)(*v58 + 16) > 0x17u && !sub_15CC8F0(*(_QWORD *)(v28 + 216), v29[5], v11) )
      {
        if ( *((_BYTE *)v29 + 16) != 56 )
          return v56;
        v30 = 24LL * (*((_DWORD *)v29 + 5) & 0xFFFFFFF);
        if ( (*((_BYTE *)v29 + 23) & 0x40) != 0 )
        {
          v31 = (__int64 *)*(v29 - 1);
          v51 = &v31[(unsigned __int64)v30 / 8];
        }
        else
        {
          v51 = v29;
          v31 = &v29[v30 / 0xFFFFFFFFFFFFFFF8LL];
        }
        if ( v31 != v51 )
          break;
      }
LABEL_70:
      v58 += 3;
      if ( v52 == v58 )
      {
        v19 = v28;
        v24 = v48;
        goto LABEL_72;
      }
    }
    v32 = v28;
    while ( 1 )
    {
      v33 = (__int64 *)*v31;
      if ( *(_BYTE *)(*v31 + 16) > 0x17u && !sub_15CC8F0(*(_QWORD *)(v32 + 216), v33[5], v11) )
      {
        if ( *((_BYTE *)v33 + 16) != 56 )
          return v56;
        v34 = 24LL * (*((_DWORD *)v33 + 5) & 0xFFFFFFF);
        if ( (*((_BYTE *)v33 + 23) & 0x40) != 0 )
        {
          v35 = (__int64 *)*(v33 - 1);
          v50 = &v35[(unsigned __int64)v34 / 8];
        }
        else
        {
          v50 = v33;
          v35 = &v33[v34 / 0xFFFFFFFFFFFFFFF8LL];
        }
        if ( v35 != v50 )
          break;
      }
LABEL_68:
      v31 += 3;
      if ( v51 == v31 )
      {
        v28 = v32;
        goto LABEL_70;
      }
    }
    v47 = v31;
    v36 = v32;
    v37 = v35;
    while ( 1 )
    {
      v38 = (__int64 *)*v37;
      if ( *(_BYTE *)(*v37 + 16) > 0x17u && !sub_15CC8F0(*(_QWORD *)(v36 + 216), v38[5], v11) )
      {
        if ( *((_BYTE *)v38 + 16) != 56 )
          return v56;
        v39 = 24LL * (*((_DWORD *)v38 + 5) & 0xFFFFFFF);
        if ( (*((_BYTE *)v38 + 23) & 0x40) != 0 )
        {
          v40 = (__int64 *)*(v38 - 1);
          v41 = &v40[(unsigned __int64)v39 / 8];
        }
        else
        {
          v41 = v38;
          v40 = &v38[v39 / 0xFFFFFFFFFFFFFFF8LL];
        }
        if ( v40 != v41 )
          break;
      }
LABEL_66:
      v37 += 3;
      if ( v50 == v37 )
      {
        v31 = v47;
        v32 = v36;
        goto LABEL_68;
      }
    }
    v46 = v37;
    v42 = v36;
    v43 = v41;
    while ( 1 )
    {
      v44 = *v40;
      if ( *(_BYTE *)(*v40 + 16) > 0x17u
        && !sub_15CC8F0(*(_QWORD *)(v42 + 216), *(_QWORD *)(v44 + 40), v11)
        && (*(_BYTE *)(v44 + 16) != 56 || !(unsigned __int8)sub_1921D00(v42, v44, v11)) )
      {
        break;
      }
      v40 += 3;
      if ( v43 == v40 )
      {
        v36 = v42;
        v37 = v46;
        goto LABEL_66;
      }
    }
  }
  return v56;
}
