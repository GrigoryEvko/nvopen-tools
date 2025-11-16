// Function: sub_2CE1D30
// Address: 0x2ce1d30
//
void __fastcall sub_2CE1D30(unsigned __int64 a1, _QWORD *a2, _QWORD *a3, __int64 a4)
{
  __int64 v6; // r12
  unsigned __int64 v7; // rbx
  _QWORD *v8; // rax
  __int64 v9; // rdi
  _QWORD *v10; // rsi
  __int64 v11; // rdx
  char v12; // al
  _QWORD *v13; // rax
  _QWORD *v14; // r15
  _QWORD *v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rdx
  _QWORD *v18; // rax
  _QWORD *v19; // rdx
  char v20; // bl
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  __int64 v23; // rsi
  _BYTE *v24; // r8
  _BYTE **v25; // rbx
  _BYTE *v26; // r15
  _QWORD *v27; // rax
  _QWORD *v28; // rdx
  _QWORD *v29; // r12
  char v30; // r15
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // r15
  _BYTE *v34; // rdx
  unsigned __int64 *v35; // rcx
  unsigned __int64 v36; // rax
  _QWORD *v37; // rax
  _QWORD *v38; // rsi
  _QWORD *v39; // rax
  _QWORD *v40; // rdx
  char v41; // r15
  __int64 v42; // rax
  _QWORD *v43; // r8
  __int64 v44; // r15
  unsigned __int8 v45; // al
  _QWORD *v46; // r10
  __int64 v47; // rdx
  __int64 v48; // r9
  __int64 v49; // rsi
  int v50; // r9d
  __int64 v51; // rax
  _QWORD *v52; // rdx
  char v53; // al
  int v54; // r9d
  unsigned __int64 v55; // rdx
  _QWORD *v56; // [rsp+0h] [rbp-90h]
  _BYTE *v57; // [rsp+8h] [rbp-88h]
  _BYTE *v58; // [rsp+8h] [rbp-88h]
  _QWORD *v59; // [rsp+10h] [rbp-80h]
  _QWORD *v60; // [rsp+10h] [rbp-80h]
  unsigned __int64 *v61; // [rsp+10h] [rbp-80h]
  _QWORD *v62; // [rsp+10h] [rbp-80h]
  _QWORD *v63; // [rsp+10h] [rbp-80h]
  _QWORD *v64; // [rsp+10h] [rbp-80h]
  unsigned __int64 v65[2]; // [rsp+18h] [rbp-78h] BYREF
  _BYTE *v66; // [rsp+28h] [rbp-68h] BYREF
  _QWORD *v67; // [rsp+30h] [rbp-60h] BYREF
  __int64 v68; // [rsp+38h] [rbp-58h]
  _QWORD v69[10]; // [rsp+40h] [rbp-50h] BYREF

  v6 = a4;
  v7 = a1;
  v8 = *(_QWORD **)(a4 + 16);
  v65[0] = a1;
  if ( !v8 )
    goto LABEL_8;
  v9 = a4 + 8;
  v10 = (_QWORD *)(a4 + 8);
  do
  {
    while ( 1 )
    {
      a4 = v8[2];
      v11 = v8[3];
      if ( v8[4] >= v7 )
        break;
      v8 = (_QWORD *)v8[3];
      if ( !v11 )
        goto LABEL_6;
    }
    v10 = v8;
    v8 = (_QWORD *)v8[2];
  }
  while ( a4 );
LABEL_6:
  if ( (_QWORD *)v9 != v10 && v10[4] <= v7 )
  {
    v27 = sub_23FDE00((__int64)a2, v65);
    v29 = v28;
    if ( v28 )
    {
      v30 = 1;
      if ( !v27 && v28 != a2 + 1 )
        v30 = v28[4] > v7;
      v31 = sub_22077B0(0x28u);
      *(_QWORD *)(v31 + 32) = v65[0];
      sub_220F040(v30, v31, v29, a2 + 1);
      ++a2[5];
    }
  }
  else
  {
LABEL_8:
    v12 = *(_BYTE *)v7;
    if ( *(_BYTE *)v7 <= 0x1Cu )
      return;
    if ( (v12 & 0xFD) != 0x54 )
    {
      if ( v12 == 78 )
      {
        sub_2CE1D30(*(_QWORD *)(v7 - 32), a2, a3, v6);
        return;
      }
      if ( v12 != 63 )
      {
        if ( v12 == 93 )
        {
          v43 = v69;
          v69[0] = v7;
          v67 = v69;
          v68 = 0x400000001LL;
          v44 = *(_QWORD *)(v7 - 32);
          v45 = *(_BYTE *)v44;
          if ( *(_BYTE *)v44 != 93 )
          {
            v46 = v69;
            v47 = 1;
            LODWORD(v48) = 1;
            goto LABEL_77;
          }
          v51 = 1;
          v52 = v69;
          while ( 1 )
          {
            v52[v51] = v44;
            v48 = (unsigned int)(v68 + 1);
            LODWORD(v68) = v68 + 1;
            v44 = *(_QWORD *)(v44 - 32);
            v45 = *(_BYTE *)v44;
            if ( *(_BYTE *)v44 != 93 )
              break;
            v55 = (unsigned int)v48 + 1LL;
            if ( v55 > HIDWORD(v68) )
            {
              v64 = v43;
              sub_C8D5F0((__int64)&v67, v43, v55, 8u, (__int64)v43, v48);
              v43 = v64;
            }
            v52 = v67;
            v51 = (unsigned int)v68;
          }
          v47 = (unsigned int)v48;
          v46 = v67;
          if ( (_DWORD)v48 )
          {
            while ( 1 )
            {
LABEL_77:
              if ( v45 <= 0x1Cu )
                goto LABEL_80;
              v49 = v46[v47 - 1];
              if ( v45 == 61 )
                break;
              if ( v45 != 94 || *(_QWORD *)(*(_QWORD *)(v49 - 32) + 8LL) != *(_QWORD *)(v44 + 8) )
                goto LABEL_80;
              v63 = v43;
              v53 = sub_2CDD660(v44, v49, v47, a4, (unsigned int)v43);
              v43 = v63;
              if ( v53 )
              {
                LODWORD(v48) = v54 - 1;
                LODWORD(v68) = v48;
                v44 = *(_QWORD *)(v44 - 32);
              }
              else
              {
                v44 = *(_QWORD *)(v44 - 64);
                LODWORD(v48) = v68;
              }
              v47 = (unsigned int)v48;
              if ( !(_DWORD)v48 )
                goto LABEL_87;
              v45 = *(_BYTE *)v44;
            }
            if ( *(_QWORD *)(*(_QWORD *)(v49 - 32) + 8LL) != *(_QWORD *)(v44 + 8) )
            {
LABEL_80:
              v50 = v68;
              goto LABEL_81;
            }
            v50 = v48 - 1;
            LODWORD(v68) = v50;
LABEL_81:
            if ( v50 )
            {
              if ( v46 != v43 )
                _libc_free((unsigned __int64)v46);
              return;
            }
          }
LABEL_87:
          v62 = v43;
          sub_2CE1D30(v44, a2, a3, v6);
          if ( v67 != v62 )
            _libc_free((unsigned __int64)v67);
          return;
        }
        if ( v12 != 85 )
          return;
        v32 = *(_QWORD *)(v7 - 32);
        if ( !v32
          || *(_BYTE *)v32
          || *(_QWORD *)(v32 + 24) != *(_QWORD *)(v7 + 80)
          || (*(_BYTE *)(v32 + 33) & 0x20) == 0
          || *(_DWORD *)(v32 + 36) != 8170 )
        {
          return;
        }
      }
      sub_2CE1D30(*(_QWORD *)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF)), a2, a3, v6);
      return;
    }
    v13 = (_QWORD *)a3[2];
    v14 = a3 + 1;
    if ( !v13 )
      goto LABEL_17;
    v15 = a3 + 1;
    do
    {
      while ( 1 )
      {
        v16 = v13[2];
        v17 = v13[3];
        if ( v13[4] >= v7 )
          break;
        v13 = (_QWORD *)v13[3];
        if ( !v17 )
          goto LABEL_15;
      }
      v15 = v13;
      v13 = (_QWORD *)v13[2];
    }
    while ( v16 );
LABEL_15:
    if ( v14 == v15 || v15[4] > v7 )
    {
LABEL_17:
      v18 = sub_23FDE00((__int64)a3, v65);
      if ( v19 )
      {
        v20 = v18 || v14 == v19 || v7 < v19[4];
        v59 = v19;
        v21 = sub_22077B0(0x28u);
        *(_QWORD *)(v21 + 32) = v65[0];
        sub_220F040(v20, v21, v59, a3 + 1);
        ++a3[5];
        v7 = v65[0];
      }
      v67 = 0;
      v68 = 0;
      v69[0] = 0;
      if ( *(_BYTE *)v7 == 84 )
      {
        if ( (*(_DWORD *)(v7 + 4) & 0x7FFFFFF) == 0 )
          return;
        v33 = 0;
        v34 = 0;
        v35 = (unsigned __int64 *)&v66;
        v24 = 0;
        while ( 1 )
        {
          v36 = *(_QWORD *)(*(_QWORD *)(v7 - 8) + 32 * v33);
          v66 = (_BYTE *)v36;
          if ( v24 == v34 )
          {
            v61 = v35;
            sub_928380((__int64)&v67, v24, v35);
            v24 = (_BYTE *)v68;
            v35 = v61;
          }
          else
          {
            if ( v24 )
            {
              *(_QWORD *)v24 = v36;
              v24 = (_BYTE *)v68;
            }
            v24 += 8;
            v68 = (__int64)v24;
          }
          if ( (*(_DWORD *)(v7 + 4) & 0x7FFFFFFu) <= (unsigned int)++v33 )
            break;
          v34 = (_BYTE *)v69[0];
        }
      }
      else
      {
        v66 = *(_BYTE **)(v7 - 64);
        sub_928380((__int64)&v67, 0, &v66);
        v22 = *(_QWORD *)(v7 - 32);
        v23 = v68;
        v66 = (_BYTE *)v22;
        if ( v69[0] == v68 )
        {
          sub_928380((__int64)&v67, (_BYTE *)v68, &v66);
          v24 = (_BYTE *)v68;
        }
        else
        {
          if ( v68 )
          {
            *(_QWORD *)v68 = v22;
            v23 = v68;
          }
          v24 = (_BYTE *)(v23 + 8);
          v68 = v23 + 8;
        }
      }
      v25 = (_BYTE **)v67;
      v60 = a2 + 1;
      if ( v67 != (_QWORD *)v24 )
      {
        while ( 1 )
        {
          v26 = *v25;
          v66 = v26;
          if ( *v26 <= 0x1Cu || (*v26 & 0xFD) != 0x54 )
          {
            if ( sub_2CDF320((__int64)v26) )
              goto LABEL_31;
            v37 = (_QWORD *)a2[2];
            if ( v37 )
            {
              v38 = a2 + 1;
              do
              {
                if ( v37[4] < (unsigned __int64)v26 )
                {
                  v37 = (_QWORD *)v37[3];
                }
                else
                {
                  v38 = v37;
                  v37 = (_QWORD *)v37[2];
                }
              }
              while ( v37 );
              if ( v60 != v38 && v38[4] <= (unsigned __int64)v26 )
                goto LABEL_31;
            }
            v58 = v24;
            v39 = sub_23FDE00((__int64)a2, (unsigned __int64 *)&v66);
            v24 = v58;
            if ( v40 )
            {
              v41 = v39 || v60 == v40 || (unsigned __int64)v26 < v40[4];
              v56 = v40;
              v42 = sub_22077B0(0x28u);
              *(_QWORD *)(v42 + 32) = v66;
              sub_220F040(v41, v42, v56, v60);
              ++a2[5];
              v26 = v66;
              v24 = v58;
            }
          }
          v57 = v24;
          sub_2CE1D30(v26, a2, a3, v6);
          v24 = v57;
LABEL_31:
          if ( v24 == (_BYTE *)++v25 )
          {
            v24 = v67;
            break;
          }
        }
      }
      if ( v24 )
        j_j___libc_free_0((unsigned __int64)v24);
    }
  }
}
