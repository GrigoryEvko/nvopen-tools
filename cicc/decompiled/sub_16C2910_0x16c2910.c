// Function: sub_16C2910
// Address: 0x16c2910
//
unsigned __int64 __fastcall sub_16C2910(
        unsigned __int64 a1,
        __int128 *a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        __int64 a6,
        char a7,
        unsigned int a8)
{
  unsigned int v11; // r12d
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r11
  __int64 v16; // rbx
  int v17; // eax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  char *v21; // r15
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  int *v25; // rbx
  int v26; // r14d
  ssize_t v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // rax
  __int64 v32; // rax
  int v34; // eax
  __int64 v35; // rbx
  int v36; // eax
  int v37; // eax
  __int64 v38; // rbx
  int v39; // eax
  __int64 v40; // rax
  __int64 v41; // rax
  int v42; // eax
  __int64 v43; // [rsp+8h] [rbp-B8h]
  __int64 v44; // [rsp+10h] [rbp-B0h]
  unsigned int v45; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v46; // [rsp+28h] [rbp-98h]
  _QWORD v47[2]; // [rsp+30h] [rbp-90h] BYREF
  char v48; // [rsp+40h] [rbp-80h]
  __int128 v49; // [rsp+50h] [rbp-70h] BYREF
  __int128 v50; // [rsp+60h] [rbp-60h]
  __int128 v51; // [rsp+70h] [rbp-50h]
  __int128 v52; // [rsp+80h] [rbp-40h]

  v11 = (unsigned int)a2;
  v13 = a8;
  v46 = a1;
  if ( !byte_4FA04E8 )
  {
    a1 = (unsigned __int64)&byte_4FA04E8;
    v34 = sub_2207590(&byte_4FA04E8);
    v13 = a8;
    if ( v34 )
    {
      a1 = (unsigned __int64)&byte_4FA04E8;
      dword_4FA04F0 = sub_16C6880(&byte_4FA04E8, a2, a3, a8);
      sub_2207640(&byte_4FA04E8);
      v13 = a8;
    }
  }
  if ( a5 == -1 )
  {
    a5 = a4;
    if ( a4 == -1 )
    {
      a1 = (unsigned int)a2;
      v45 = v13;
      a2 = &v49;
      v51 = 0;
      v49 = 0;
      DWORD1(v51) = 0xFFFF;
      v50 = 0;
      v52 = 0;
      v37 = sub_16C5920(a1);
      v13 = v45;
      if ( v37 )
      {
        *(_BYTE *)(v46 + 16) |= 1u;
        *(_DWORD *)v46 = v37;
        *(_QWORD *)(v46 + 8) = a3;
        return v46;
      }
      if ( (_DWORD)v51 != 2 && (_DWORD)v51 != 5 )
      {
        sub_16C2770((__int64)v47, v11, a3);
        if ( (v48 & 1) != 0 )
        {
          v42 = v47[0];
          *(_BYTE *)(v46 + 16) |= 1u;
          *(_DWORD *)v46 = v42;
          *(_QWORD *)(v46 + 8) = v47[1];
        }
        else
        {
          v41 = v47[0];
          *(_BYTE *)(v46 + 16) &= ~1u;
          *(_QWORD *)v46 = v41;
        }
        return v46;
      }
      a4 = *((_QWORD *)&v50 + 1);
      a5 = *((_QWORD *)&v50 + 1);
    }
  }
  if ( a5 > 0x3FFF && !(_BYTE)v13 )
  {
    v14 = (unsigned int)dword_4FA04F0;
    if ( a5 >= (unsigned int)dword_4FA04F0 )
    {
      if ( !a7 || (a2 = (__int128 *)a4, a1 = v11, sub_16C2170(v11, a4, a5, a6, dword_4FA04F0)) )
      {
        LODWORD(v49) = 0;
        *((_QWORD *)&v49 + 1) = sub_2241E40(a1, a2, a3, v13, v14);
        v15 = sub_16C2200(48, a3);
        if ( v15 )
        {
          v16 = v15 + 24;
          v43 = v15;
          *(_QWORD *)v15 = off_49850B0;
          v44 = a6 & (int)-sub_16C5A70();
          v17 = sub_16C5A70();
          sub_16C59F0(v16, v11, 0, a5 + (a6 & (v17 - 1)), v44);
          v15 = v43;
          if ( !(_DWORD)v49 )
          {
            v35 = sub_16C5A60(v16);
            v36 = sub_16C5A70();
            sub_16C2440(v43, v35 + (a6 & (v36 - 1)), v35 + (a6 & (v36 - 1)) + a5);
            v15 = v43;
            if ( !(_DWORD)v49 )
              goto LABEL_27;
          }
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v15 + 8LL))(v15);
        }
        else if ( !(_DWORD)v49 )
        {
LABEL_27:
          *(_BYTE *)(v46 + 16) &= ~1u;
          *(_QWORD *)v46 = v15;
          return v46;
        }
      }
    }
  }
  sub_16C2500(&v49, a5, a3);
  if ( (_QWORD)v49 )
  {
    v21 = *(char **)(v49 + 8);
    if ( lseek(v11, a6, 0) == -1 )
    {
      v38 = sub_2241E50(v11, a6, v22, v23, v24);
      v39 = *__errno_location();
      *(_BYTE *)(v46 + 16) |= 1u;
      *(_DWORD *)v46 = v39;
      *(_QWORD *)(v46 + 8) = v38;
LABEL_31:
      if ( (_QWORD)v49 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v49 + 8LL))(v49);
    }
    else
    {
      if ( a5 )
      {
        v25 = __errno_location();
        while ( 1 )
        {
          while ( 1 )
          {
            *v25 = 0;
            v27 = read(v11, v21, a5);
            if ( v27 != -1 )
              break;
            v26 = *v25;
            if ( *v25 != 4 )
            {
              v40 = sub_2241E50(v11, v21, v28, v29, v30);
              *(_BYTE *)(v46 + 16) |= 1u;
              *(_DWORD *)v46 = v26;
              *(_QWORD *)(v46 + 8) = v40;
              goto LABEL_31;
            }
          }
          if ( !v27 )
            break;
          v21 += v27;
          a5 -= v27;
          if ( !a5 )
            goto LABEL_21;
        }
        memset(v21, 0, a5);
      }
LABEL_21:
      v31 = v49;
      *(_BYTE *)(v46 + 16) &= ~1u;
      *(_QWORD *)v46 = v31;
    }
  }
  else
  {
    v32 = sub_2241E50(&v49, a5, v18, v19, v20);
    *(_BYTE *)(v46 + 16) |= 1u;
    *(_DWORD *)v46 = 12;
    *(_QWORD *)(v46 + 8) = v32;
  }
  return v46;
}
