// Function: sub_221B7A0
// Address: 0x221b7a0
//
__int64 __fastcall sub_221B7A0(__int64 a1, __int64 a2, char a3, __int64 a4, char a5, __int64 a6)
{
  int v7; // r14d
  __int64 v9; // r12
  __int64 v10; // rbp
  const char *v11; // r12
  __int64 v12; // rdx
  unsigned __int8 *v13; // rdx
  const char *v14; // r14
  unsigned __int64 v15; // r14
  __int64 result; // rax
  __int64 v17; // rdx
  __int64 v18; // rbx
  int v19; // eax
  _BYTE *v20; // rax
  unsigned __int64 v21; // r13
  unsigned __int64 *v22; // rbx
  __int64 v23; // rax
  __int64 v24; // kr00_8
  unsigned __int64 v25; // r13
  _QWORD *v26; // rdx
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rdx
  const char *v29; // rsi
  unsigned __int64 v30; // r13
  _QWORD *v31; // rdx
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rsi
  _QWORD *v34; // rsi
  unsigned __int64 v35; // r13
  unsigned __int64 v36; // rdx
  unsigned __int64 v37; // rax
  __int64 v38; // rbp
  int v39; // edx
  int v40; // ecx
  int v41; // r8d
  int v42; // r9d
  __int64 v44; // [rsp+8h] [rbp-100h]
  __int64 *v45; // [rsp+10h] [rbp-F8h]
  unsigned __int64 v46; // [rsp+10h] [rbp-F8h]
  unsigned __int64 v49; // [rsp+28h] [rbp-E0h]
  char v50; // [rsp+28h] [rbp-E0h]
  char *v51; // [rsp+30h] [rbp-D8h]
  int v52; // [rsp+38h] [rbp-D0h]
  unsigned __int64 v53; // [rsp+38h] [rbp-D0h]
  bool v55; // [rsp+53h] [rbp-B5h]
  unsigned __int64 v56; // [rsp+58h] [rbp-B0h]
  char v57; // [rsp+60h] [rbp-A8h]
  unsigned __int64 v58; // [rsp+60h] [rbp-A8h]
  unsigned __int64 v59; // [rsp+68h] [rbp-A0h]
  int v60; // [rsp+8Ch] [rbp-7Ch] BYREF
  _QWORD *v61; // [rsp+90h] [rbp-78h] BYREF
  unsigned __int64 v62; // [rsp+98h] [rbp-70h]
  _QWORD v63[2]; // [rsp+A0h] [rbp-68h] BYREF
  _QWORD *v64; // [rsp+B0h] [rbp-58h] BYREF
  unsigned __int64 v65; // [rsp+B8h] [rbp-50h]
  _QWORD v66[9]; // [rsp+C0h] [rbp-48h] BYREF

  v7 = a4 + 208;
  v44 = sub_222F790(a4 + 208);
  v9 = sub_22091A0(&qword_4FD6888);
  v10 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a4 + 208) + 24LL) + 8 * v9);
  v45 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a4 + 208) + 24LL) + 8 * v9);
  if ( !v10 )
  {
    v38 = sub_22077B0(0x70u);
    *(_DWORD *)(v38 + 8) = 0;
    *(_QWORD *)(v38 + 16) = 0;
    *(_QWORD *)(v38 + 24) = 0;
    *(_QWORD *)v38 = off_4A04880;
    *(_WORD *)(v38 + 32) = 0;
    *(_BYTE *)(v38 + 34) = 0;
    *(_QWORD *)(v38 + 40) = 0;
    *(_QWORD *)(v38 + 48) = 0;
    *(_QWORD *)(v38 + 56) = 0;
    *(_QWORD *)(v38 + 64) = 0;
    *(_QWORD *)(v38 + 72) = 0;
    *(_QWORD *)(v38 + 80) = 0;
    *(_QWORD *)(v38 + 88) = 0;
    *(_DWORD *)(v38 + 96) = 0;
    *(_BYTE *)(v38 + 111) = 0;
    sub_2230AE0(v38, v7, v39, v40, v41, v42);
    sub_2209690(*(_QWORD *)(a4 + 208), (volatile signed __int32 *)v38, v9);
    v10 = *v45;
  }
  v11 = *(const char **)a6;
  v12 = *(_QWORD *)(a6 + 8);
  if ( **(_BYTE **)a6 == *(_BYTE *)(v10 + 100) )
  {
    v60 = *(_DWORD *)(v10 + 96);
    if ( !v12 )
      goto LABEL_10;
    ++v11;
    v51 = *(char **)(v10 + 72);
    v46 = *(_QWORD *)(v10 + 80);
  }
  else
  {
    v60 = *(_DWORD *)(v10 + 92);
    v51 = *(char **)(v10 + 56);
    v46 = *(_QWORD *)(v10 + 64);
  }
  v13 = (unsigned __int8 *)&v11[v12];
  if ( v13 > (unsigned __int8 *)v11 )
  {
    v14 = v11;
    do
    {
      if ( (*(_BYTE *)(*(_QWORD *)(v44 + 48) + 2LL * *(unsigned __int8 *)v14 + 1) & 8) == 0 )
        break;
      ++v14;
    }
    while ( v13 != (unsigned __int8 *)v14 );
    v15 = v14 - v11;
    if ( v15 )
    {
      v62 = 0;
      v61 = v63;
      LOBYTE(v63[0]) = 0;
      sub_2240E30(&v61, 2 * v15);
      v17 = *(int *)(v10 + 88);
      v18 = v15 - v17;
      v19 = *(_DWORD *)(v10 + 88);
      if ( (__int64)(v15 - v17) > 0 )
      {
        if ( (int)v17 < 0 )
          v18 = v15;
        if ( *(_QWORD *)(v10 + 24) )
        {
          sub_2240FD0(&v61, 0, v62, 2 * v18, 0);
          v20 = (_BYTE *)sub_2231480(
                           v61,
                           (unsigned int)*(char *)(v10 + 34),
                           *(_QWORD *)(v10 + 16),
                           *(_QWORD *)(v10 + 24),
                           v11,
                           &v11[v18]);
          if ( v20 - (_BYTE *)v61 > v62 )
            sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::erase");
          v62 = v20 - (_BYTE *)v61;
          *v20 = 0;
          v19 = *(_DWORD *)(v10 + 88);
        }
        else
        {
          sub_2241130(&v61, 0, v62, v11, v18);
          v19 = *(_DWORD *)(v10 + 88);
        }
      }
      if ( v19 > 0 )
      {
        v35 = v62;
        v36 = (unsigned __int64)v61;
        v50 = *(_BYTE *)(v10 + 33);
        v37 = 15;
        if ( v61 != v63 )
          v37 = v63[0];
        v53 = v62 + 1;
        if ( v62 + 1 > v37 )
        {
          sub_2240BB0(&v61, v62, 0, 0, 1);
          v36 = (unsigned __int64)v61;
        }
        *(_BYTE *)(v36 + v35) = v50;
        v62 = v53;
        *((_BYTE *)v61 + v35 + 1) = 0;
        if ( v18 < 0 )
        {
          sub_2240FD0(&v61, v62, 0, -v18, (unsigned int)*(char *)(v10 + 101));
          if ( v15 > 0x3FFFFFFFFFFFFFFFLL - v62 )
            sub_4262D8((__int64)"basic_string::append");
          sub_2241490(&v61, v11);
        }
        else
        {
          if ( *(int *)(v10 + 88) > 0x3FFFFFFFFFFFFFFFLL - v62 )
            sub_4262D8((__int64)"basic_string::append");
          sub_2241490(&v61, &v11[v18]);
        }
      }
      v21 = v62 + v46;
      v52 = *(_DWORD *)(a4 + 24) & 0xB0;
      if ( (*(_DWORD *)(a4 + 24) & 0x200) != 0 )
        v21 += *(_QWORD *)(v10 + 48);
      v65 = 0;
      v64 = v66;
      LOBYTE(v66[0]) = 0;
      sub_2240E30(&v64, 2 * v21);
      v22 = (unsigned __int64 *)&v60;
      v49 = *(_QWORD *)(a4 + 16);
      v23 = (unsigned int)a5;
      v56 = v49 - v21;
      v55 = v21 < v49 && v52 == 16;
      do
      {
        v24 = v23;
        v23 = *(unsigned __int8 *)v22;
        switch ( *(_BYTE *)v22 )
        {
          case 0:
            if ( v55 )
              goto LABEL_40;
            break;
          case 1:
            v30 = v65;
            if ( v55 )
            {
LABEL_40:
              v23 = sub_2240FD0(&v64, v65, 0, v56, (unsigned int)a5);
            }
            else
            {
              v31 = v64;
              v32 = 15;
              if ( v64 != v66 )
                v32 = v66[0];
              v58 = v65 + 1;
              if ( v65 + 1 > v32 )
              {
                sub_2240BB0(&v64, v65, 0, 0, 1);
                v31 = v64;
              }
              *((_BYTE *)v31 + v30) = a5;
              v65 = v58;
              v23 = (__int64)v64;
              *((_BYTE *)v64 + v30 + 1) = 0;
            }
            break;
          case 2:
            if ( (*(_BYTE *)(a4 + 25) & 2) != 0 )
            {
              v28 = *(_QWORD *)(v10 + 48);
              v29 = *(const char **)(v10 + 40);
              if ( v28 > 0x3FFFFFFFFFFFFFFFLL - v65 )
                sub_4262D8((__int64)"basic_string::append");
              goto LABEL_42;
            }
            return result;
          case 3:
            if ( v46 )
            {
              v25 = v65;
              v26 = v64;
              v59 = v65 + 1;
              v57 = *v51;
              v27 = 15;
              if ( v64 != v66 )
                v27 = v66[0];
              if ( v65 + 1 > v27 )
              {
                sub_2240BB0(&v64, v65, 0, 0, 1);
                v26 = v64;
              }
              *((_BYTE *)v26 + v25) = v57;
              v65 = v59;
              v23 = (__int64)v64;
              *((_BYTE *)v64 + v25 + 1) = 0;
            }
            break;
          case 4:
            v28 = v62;
            v29 = (const char *)v61;
LABEL_42:
            v23 = sub_2241490(&v64, v29, v28);
            break;
          default:
            v23 = v24;
            break;
        }
        v22 = (unsigned __int64 *)((char *)v22 + 1);
      }
      while ( v22 != (unsigned __int64 *)&v61 );
      if ( v46 > 1 )
      {
        if ( v46 - 1 > 0x3FFFFFFFFFFFFFFFLL - v65 )
          sub_4262D8((__int64)"basic_string::append");
        sub_2241490(&v64, v51 + 1);
      }
      v33 = v65;
      if ( v49 <= v65 )
      {
        LODWORD(v49) = v65;
      }
      else
      {
        if ( v52 != 32 )
          v33 = 0;
        sub_2240FD0(&v64, v33, 0, v49 - v65, (unsigned int)a5);
      }
      v34 = v64;
      if ( !a3 )
      {
        (*(void (__fastcall **)(__int64, _QWORD *, _QWORD))(*(_QWORD *)a2 + 96LL))(a2, v64, (int)v49);
        v34 = v64;
      }
      if ( v34 != v66 )
        j___libc_free_0((unsigned __int64)v34);
      if ( v61 != v63 )
        j___libc_free_0((unsigned __int64)v61);
    }
  }
LABEL_10:
  *(_QWORD *)(a4 + 16) = 0;
  return a2;
}
