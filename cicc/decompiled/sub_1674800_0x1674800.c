// Function: sub_1674800
// Address: 0x1674800
//
__int64 __fastcall sub_1674800(__int64 a1, _QWORD **a2, __int64 a3)
{
  unsigned int v6; // esi
  __int64 v7; // rdi
  unsigned int v8; // ecx
  _QWORD *v9; // rdx
  __int64 v10; // rax
  __int64 result; // rax
  int v12; // r10d
  _QWORD *v13; // r14
  int v14; // eax
  int v15; // edx
  unsigned int v16; // r8d
  _QWORD *v17; // rax
  char v18; // dl
  unsigned __int64 v19; // r14
  char v20; // bl
  __int64 **v21; // rax
  __int64 **i; // rdx
  int v23; // eax
  __int64 **v24; // rdx
  _QWORD *v25; // rax
  int v26; // r15d
  bool v27; // si
  unsigned int v28; // esi
  __int64 v29; // rdi
  unsigned int v30; // ecx
  _QWORD *v31; // r15
  __int64 v32; // rdx
  int v33; // eax
  int v34; // ecx
  __int64 v35; // rsi
  unsigned int v36; // eax
  __int64 v37; // rdi
  int v38; // r10d
  _QWORD *v39; // r8
  int v40; // eax
  int v41; // eax
  __int64 v42; // rsi
  int v43; // r8d
  unsigned int v44; // ebx
  _QWORD *v45; // rdi
  _QWORD **v46; // rcx
  _QWORD *v47; // rsi
  unsigned int v48; // edi
  _QWORD *v49; // rcx
  int v50; // r9d
  int v51; // edi
  int v52; // ecx
  unsigned int v53; // edx
  unsigned int v54; // eax
  unsigned int v55; // edx
  char v56; // al
  __int64 v57; // rdi
  int v58; // eax
  int v59; // esi
  __int64 v60; // rdx
  unsigned int v61; // eax
  __int64 v62; // rdi
  int v63; // r9d
  _QWORD *v64; // r8
  int v65; // edx
  int v66; // edx
  __int64 v67; // rdi
  int v68; // r9d
  unsigned int v69; // eax
  __int64 v70; // rsi
  __int64 v71; // rax
  __int64 v72; // [rsp+10h] [rbp-80h]
  char v73; // [rsp+1Bh] [rbp-75h]
  int v74; // [rsp+1Ch] [rbp-74h]
  __int64 v75; // [rsp+20h] [rbp-70h]
  __int64 v76; // [rsp+28h] [rbp-68h]
  unsigned int v77; // [rsp+28h] [rbp-68h]
  unsigned int v78; // [rsp+28h] [rbp-68h]
  __int64 v79; // [rsp+28h] [rbp-68h]
  __int64 **v80; // [rsp+28h] [rbp-68h]
  __int64 **v81; // [rsp+30h] [rbp-60h] BYREF
  __int64 v82; // [rsp+38h] [rbp-58h]
  _BYTE v83[80]; // [rsp+40h] [rbp-50h] BYREF

  v6 = *(_DWORD *)(a1 + 32);
  v72 = a1 + 8;
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_43;
  }
  v7 = *(_QWORD *)(a1 + 16);
  v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (_QWORD *)(v7 + 16LL * v8);
  v10 = *v9;
  if ( (_QWORD **)*v9 != a2 )
  {
    v12 = 1;
    v13 = 0;
    while ( v10 != -8 )
    {
      if ( v10 == -16 && !v13 )
        v13 = v9;
      v8 = (v6 - 1) & (v12 + v8);
      v9 = (_QWORD *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( (_QWORD **)*v9 == a2 )
        goto LABEL_3;
      ++v12;
    }
    v14 = *(_DWORD *)(a1 + 24);
    if ( !v13 )
      v13 = v9;
    ++*(_QWORD *)(a1 + 8);
    v15 = v14 + 1;
    if ( 4 * (v14 + 1) < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(a1 + 28) - v15 > v6 >> 3 )
      {
LABEL_11:
        *(_DWORD *)(a1 + 24) = v15;
        if ( *v13 != -8 )
          --*(_DWORD *)(a1 + 28);
        *v13 = a2;
        v13[1] = 0;
        goto LABEL_14;
      }
      sub_1670A20(v72, v6);
      v40 = *(_DWORD *)(a1 + 32);
      if ( v40 )
      {
        v41 = v40 - 1;
        v42 = *(_QWORD *)(a1 + 16);
        v43 = 1;
        v44 = v41 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v15 = *(_DWORD *)(a1 + 24) + 1;
        v45 = 0;
        v13 = (_QWORD *)(v42 + 16LL * v44);
        v46 = (_QWORD **)*v13;
        if ( (_QWORD **)*v13 != a2 )
        {
          while ( v46 != (_QWORD **)-8LL )
          {
            if ( !v45 && v46 == (_QWORD **)-16LL )
              v45 = v13;
            v44 = v41 & (v43 + v44);
            v13 = (_QWORD *)(v42 + 16LL * v44);
            v46 = (_QWORD **)*v13;
            if ( (_QWORD **)*v13 == a2 )
              goto LABEL_11;
            ++v43;
          }
          if ( v45 )
            v13 = v45;
        }
        goto LABEL_11;
      }
LABEL_139:
      ++*(_DWORD *)(a1 + 24);
      BUG();
    }
LABEL_43:
    sub_1670A20(v72, 2 * v6);
    v33 = *(_DWORD *)(a1 + 32);
    if ( v33 )
    {
      v34 = v33 - 1;
      v35 = *(_QWORD *)(a1 + 16);
      v36 = (v33 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = *(_DWORD *)(a1 + 24) + 1;
      v13 = (_QWORD *)(v35 + 16LL * v36);
      v37 = *v13;
      if ( (_QWORD **)*v13 != a2 )
      {
        v38 = 1;
        v39 = 0;
        while ( v37 != -8 )
        {
          if ( !v39 && v37 == -16 )
            v39 = v13;
          v36 = v34 & (v38 + v36);
          v13 = (_QWORD *)(v35 + 16LL * v36);
          v37 = *v13;
          if ( (_QWORD **)*v13 == a2 )
            goto LABEL_11;
          ++v38;
        }
        if ( v39 )
          v13 = v39;
      }
      goto LABEL_11;
    }
    goto LABEL_139;
  }
LABEL_3:
  result = v9[1];
  if ( result )
    return result;
  v13 = v9;
LABEL_14:
  if ( *((_BYTE *)a2 + 8) == 13 && (*((_BYTE *)a2 + 9) & 4) == 0 )
  {
    if ( (unsigned __int8)sub_16033B0((__int64)*a2)
      && (*((_BYTE *)a2 + 9) & 1) != 0
      && sub_16707E0(*(_QWORD *)(a1 + 640), (__int64)a2) )
    {
      v13[1] = a2;
      return (__int64)a2;
    }
    v17 = *(_QWORD **)(a3 + 8);
    if ( *(_QWORD **)(a3 + 16) != v17 )
      goto LABEL_23;
    v47 = &v17[*(unsigned int *)(a3 + 28)];
    v48 = *(_DWORD *)(a3 + 28);
    if ( v17 != v47 )
    {
      v49 = 0;
      do
      {
        if ( (_QWORD **)*v17 == a2 )
          goto LABEL_62;
        if ( *v17 == -2 )
          v49 = v17;
        ++v17;
      }
      while ( v47 != v17 );
      if ( v49 )
      {
        *v49 = a2;
        --*(_DWORD *)(a3 + 32);
        ++*(_QWORD *)a3;
        goto LABEL_24;
      }
    }
    if ( v48 < *(_DWORD *)(a3 + 24) )
    {
      *(_DWORD *)(a3 + 28) = v48 + 1;
      *v47 = a2;
      ++*(_QWORD *)a3;
    }
    else
    {
LABEL_23:
      sub_16CCBA0(a3, a2);
      if ( !v18 )
      {
LABEL_62:
        result = sub_16440F0((__int64)*a2);
        v13[1] = result;
        return result;
      }
    }
LABEL_24:
    v19 = *((unsigned int *)a2 + 3);
    v73 = 0;
    v81 = (__int64 **)v83;
    v16 = v19;
    v82 = 0x400000000LL;
    if ( !v19 )
      goto LABEL_25;
    goto LABEL_27;
  }
  v81 = (__int64 **)v83;
  v16 = *((_DWORD *)a2 + 3);
  v82 = 0x400000000LL;
  if ( v16 )
  {
    v73 = 1;
    v19 = v16;
LABEL_27:
    if ( v19 > 4 )
    {
      v77 = v16;
      sub_16CD150(&v81, v83, v19, 8);
      v16 = v77;
    }
    v21 = &v81[(unsigned int)v82];
    for ( i = &v81[v19]; i != v21; ++v21 )
    {
      if ( v21 )
        *v21 = 0;
    }
    v23 = *((_DWORD *)a2 + 3);
    LODWORD(v82) = v16;
    v74 = v23;
    if ( v23 )
    {
      v24 = v81;
      v20 = 0;
      v75 = a3;
      v25 = a2[2];
      v26 = 0;
      do
      {
        v24[v26] = (__int64 *)sub_1674800(a1, v25[v26], v75);
        v24 = v81;
        v25 = a2[2];
        v27 = v81[v26] != (__int64 *)v25[v26];
        ++v26;
        v20 |= v27;
      }
      while ( v26 != v74 );
LABEL_36:
      v28 = *(_DWORD *)(a1 + 32);
      if ( v28 )
      {
        v29 = *(_QWORD *)(a1 + 16);
        v30 = (v28 - 1) & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
        v31 = (_QWORD *)(v29 + 16LL * v30);
        v32 = *v31;
        if ( (_QWORD **)*v31 == a2 )
        {
LABEL_38:
          result = v31[1];
          if ( result )
          {
            if ( *(_BYTE *)(result + 8) == 13 && (*(_BYTE *)(result + 9) & 1) == 0 )
            {
              sub_1673EB0(a1, (__int64 **)v31[1], (__int64)a2, v81, (unsigned int)v82);
              result = v31[1];
            }
            goto LABEL_17;
          }
          v13 = v31;
          goto LABEL_72;
        }
        v50 = 1;
        v13 = 0;
        while ( v32 != -8 )
        {
          if ( !v13 && v32 == -16 )
            v13 = v31;
          v30 = (v28 - 1) & (v50 + v30);
          v31 = (_QWORD *)(v29 + 16LL * v30);
          v32 = *v31;
          if ( (_QWORD **)*v31 == a2 )
            goto LABEL_38;
          ++v50;
        }
        v51 = *(_DWORD *)(a1 + 24);
        if ( !v13 )
          v13 = v31;
        ++*(_QWORD *)(a1 + 8);
        v52 = v51 + 1;
        if ( 4 * (v51 + 1) < 3 * v28 )
        {
          if ( v28 - *(_DWORD *)(a1 + 28) - v52 > v28 >> 3 )
          {
LABEL_69:
            *(_DWORD *)(a1 + 24) = v52;
            if ( *v13 != -8 )
              --*(_DWORD *)(a1 + 28);
            *v13 = a2;
            v13[1] = 0;
LABEL_72:
            if ( v20 == 1 || !v73 )
            {
              switch ( *((_BYTE *)a2 + 8) )
              {
                case 0xC:
                  result = sub_1644EA0(*v81, v81 + 1, (unsigned int)v82 - 1LL, *((_DWORD *)a2 + 2) >> 8 != 0);
                  v13[1] = result;
                  break;
                case 0xD:
                  v53 = *((_DWORD *)a2 + 2);
                  v54 = v53 >> 9;
                  v55 = v53 >> 8;
                  v56 = v54 & 1;
                  if ( v73 )
                  {
                    result = sub_1645600(*a2, v81, (unsigned int)v82, v56);
                    v13[1] = result;
                  }
                  else
                  {
                    v57 = *(_QWORD *)(a1 + 640);
                    if ( (v55 & 1) != 0 )
                    {
                      v71 = sub_1670940(v57, (__int64)v81, (unsigned int)v82, v56);
                      if ( v71 )
                      {
                        if ( *(_BYTE *)(a1 + 648) && (*(_BYTE *)(v71 + 9) & 4) != 0 )
                          goto LABEL_16;
                        v79 = v71;
                        sub_1643660(a2, byte_3F871B3, 0);
                        result = v79;
                        v13[1] = v79;
                      }
                      else
                      {
                        if ( !v20 )
                        {
                          sub_1673BC0(*(_QWORD *)(a1 + 640), (__int64)a2);
                          goto LABEL_16;
                        }
                        v80 = (__int64 **)sub_16440F0((__int64)*a2);
                        sub_1673EB0(a1, v80, (__int64)a2, v81, (unsigned int)v82);
                        result = (__int64)v80;
                        v13[1] = v80;
                      }
                    }
                    else
                    {
                      sub_1674160(v57, (__int64)a2);
                      v13[1] = a2;
                      result = (__int64)a2;
                    }
                  }
                  break;
                case 0xE:
                  result = (__int64)sub_1645D80(*v81, (__int64)a2[4]);
                  v13[1] = result;
                  break;
                case 0xF:
                  result = sub_1646BA0(*v81, *((_DWORD *)a2 + 2) >> 8);
                  v13[1] = result;
                  break;
                case 0x10:
                  result = (__int64)sub_16463B0(*v81, (unsigned int)a2[4]);
                  v13[1] = result;
                  break;
              }
              goto LABEL_17;
            }
            goto LABEL_16;
          }
          v78 = ((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9);
          sub_1670A20(v72, v28);
          v65 = *(_DWORD *)(a1 + 32);
          if ( v65 )
          {
            v66 = v65 - 1;
            v67 = *(_QWORD *)(a1 + 16);
            v68 = 1;
            v64 = 0;
            v69 = v66 & v78;
            v52 = *(_DWORD *)(a1 + 24) + 1;
            v13 = (_QWORD *)(v67 + 16LL * (v66 & v78));
            v70 = *v13;
            if ( (_QWORD **)*v13 == a2 )
              goto LABEL_69;
            while ( v70 != -8 )
            {
              if ( !v64 && v70 == -16 )
                v64 = v13;
              v69 = v66 & (v68 + v69);
              v13 = (_QWORD *)(v67 + 16LL * v69);
              v70 = *v13;
              if ( (_QWORD **)*v13 == a2 )
                goto LABEL_69;
              ++v68;
            }
            goto LABEL_87;
          }
          goto LABEL_140;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 8);
      }
      sub_1670A20(v72, 2 * v28);
      v58 = *(_DWORD *)(a1 + 32);
      if ( v58 )
      {
        v59 = v58 - 1;
        v60 = *(_QWORD *)(a1 + 16);
        v61 = (v58 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v52 = *(_DWORD *)(a1 + 24) + 1;
        v13 = (_QWORD *)(v60 + 16LL * v61);
        v62 = *v13;
        if ( (_QWORD **)*v13 == a2 )
          goto LABEL_69;
        v63 = 1;
        v64 = 0;
        while ( v62 != -8 )
        {
          if ( v62 == -16 && !v64 )
            v64 = v13;
          v61 = v59 & (v63 + v61);
          v13 = (_QWORD *)(v60 + 16LL * v61);
          v62 = *v13;
          if ( (_QWORD **)*v13 == a2 )
            goto LABEL_69;
          ++v63;
        }
LABEL_87:
        if ( v64 )
          v13 = v64;
        goto LABEL_69;
      }
LABEL_140:
      ++*(_DWORD *)(a1 + 24);
      BUG();
    }
LABEL_25:
    v20 = 0;
    goto LABEL_36;
  }
LABEL_16:
  v13[1] = a2;
  result = (__int64)a2;
LABEL_17:
  if ( v81 != (__int64 **)v83 )
  {
    v76 = result;
    _libc_free((unsigned __int64)v81);
    return v76;
  }
  return result;
}
