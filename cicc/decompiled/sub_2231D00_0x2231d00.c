// Function: sub_2231D00
// Address: 0x2231d00
//
__int64 __fastcall sub_2231D00(__int64 a1, __int64 a2, char a3, __int64 a4, char a5, volatile signed __int32 **a6)
{
  _QWORD *v6; // r15
  __int64 v9; // r12
  __int64 *v10; // r14
  __int64 v11; // rbp
  volatile signed __int32 *v12; // r14
  __int64 v13; // rdx
  unsigned __int8 *v14; // rdx
  volatile signed __int32 *v15; // r12
  size_t v16; // r12
  __int64 v18; // rdx
  signed __int64 v19; // rbx
  int v20; // eax
  volatile signed __int32 *v21; // rdi
  char v22; // r13
  _BYTE *v23; // rax
  volatile signed __int32 *v24; // rdi
  _BYTE *v25; // r13
  unsigned __int64 v26; // rcx
  unsigned __int64 v27; // r14
  volatile signed __int32 **v28; // rbx
  __int64 v29; // rax
  __int64 v30; // kr00_8
  char v31; // dl
  volatile signed __int32 *v32; // rsi
  unsigned __int64 v33; // rax
  __int64 v34; // rbp
  __int64 v37; // [rsp+18h] [rbp-A0h]
  unsigned __int64 v38; // [rsp+18h] [rbp-A0h]
  unsigned __int64 v39; // [rsp+20h] [rbp-98h]
  char *v43; // [rsp+38h] [rbp-80h]
  __int64 v44; // [rsp+40h] [rbp-78h]
  int v45; // [rsp+40h] [rbp-78h]
  char *v46; // [rsp+48h] [rbp-70h]
  bool v47; // [rsp+48h] [rbp-70h]
  int v48; // [rsp+6Ch] [rbp-4Ch] BYREF
  volatile signed __int32 *v49; // [rsp+70h] [rbp-48h] BYREF
  volatile signed __int32 *v50[8]; // [rsp+78h] [rbp-40h] BYREF

  v6 = (_QWORD *)(a4 + 208);
  v37 = sub_222F790((_QWORD *)(a4 + 208), a2);
  v9 = sub_22091A0(&qword_4FD69D8);
  v10 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a4 + 208) + 24LL) + 8 * v9);
  v11 = *v10;
  if ( !*v10 )
  {
    v34 = sub_22077B0(0x70u);
    *(_DWORD *)(v34 + 8) = 0;
    *(_QWORD *)(v34 + 16) = 0;
    *(_QWORD *)(v34 + 24) = 0;
    *(_QWORD *)v34 = off_4A04880;
    *(_WORD *)(v34 + 32) = 0;
    *(_BYTE *)(v34 + 34) = 0;
    *(_QWORD *)(v34 + 40) = 0;
    *(_QWORD *)(v34 + 48) = 0;
    *(_QWORD *)(v34 + 56) = 0;
    *(_QWORD *)(v34 + 64) = 0;
    *(_QWORD *)(v34 + 72) = 0;
    *(_QWORD *)(v34 + 80) = 0;
    *(_QWORD *)(v34 + 88) = 0;
    *(_DWORD *)(v34 + 96) = 0;
    *(_BYTE *)(v34 + 111) = 0;
    sub_2230AE0(v34, v6);
    sub_2209690(*(_QWORD *)(a4 + 208), (volatile signed __int32 *)v34, v9);
    v11 = *v10;
  }
  v12 = *a6;
  v13 = *((_QWORD *)*a6 - 3);
  if ( *(_BYTE *)*a6 == *(_BYTE *)(v11 + 100) )
  {
    v48 = *(_DWORD *)(v11 + 96);
    if ( !v13 )
      goto LABEL_10;
    v12 = (volatile signed __int32 *)((char *)v12 + 1);
    v43 = *(char **)(v11 + 72);
    v39 = *(_QWORD *)(v11 + 80);
  }
  else
  {
    v48 = *(_DWORD *)(v11 + 92);
    v43 = *(char **)(v11 + 56);
    v39 = *(_QWORD *)(v11 + 64);
  }
  v14 = (unsigned __int8 *)v12 + v13;
  if ( v14 > (unsigned __int8 *)v12 )
  {
    v15 = v12;
    do
    {
      if ( (*(_BYTE *)(*(_QWORD *)(v37 + 48) + 2LL * *(unsigned __int8 *)v15 + 1) & 8) == 0 )
        break;
      v15 = (volatile signed __int32 *)((char *)v15 + 1);
    }
    while ( v14 != (unsigned __int8 *)v15 );
    v16 = (char *)v15 - (char *)v12;
    if ( v16 )
    {
      v49 = (volatile signed __int32 *)&unk_4FD67D8;
      sub_2215AB0((__int64 *)&v49, 2 * v16);
      v18 = *(int *)(v11 + 88);
      v19 = v16 - v18;
      v20 = *(_DWORD *)(v11 + 88);
      if ( (__int64)(v16 - v18) > 0 )
      {
        if ( (int)v18 < 0 )
          v19 = v16;
        if ( *(_QWORD *)(v11 + 24) )
        {
          sub_22157B0(&v49, 0, *((_QWORD *)v49 - 3), 2 * v19, 0);
          v21 = v49;
          v22 = *(_BYTE *)(v11 + 34);
          v44 = *(_QWORD *)(v11 + 24);
          v46 = *(char **)(v11 + 16);
          if ( *((int *)v49 - 2) >= 0 )
          {
            sub_2215730(&v49);
            v21 = v49;
          }
          v23 = sub_2231480(v21, v22, v46, v44, (__int64)v12, (__int64)v12 + v19);
          v24 = v49;
          v25 = v23;
          if ( *((int *)v49 - 2) >= 0 )
          {
            sub_2215730(&v49);
            v24 = v49;
          }
          v26 = *((_QWORD *)v24 - 3);
          if ( v25 - (_BYTE *)v24 > v26 )
            sub_222CF80(
              "%s: __pos (which is %zu) > this->size() (which is %zu)",
              "basic_string::erase",
              v25 - (_BYTE *)v24,
              v26);
          sub_2215540(&v49, v25 - (_BYTE *)v24, *((_QWORD *)v24 - 3) - (v25 - (_BYTE *)v24), 0);
        }
        else
        {
          sub_22158B0(&v49, v12, v19);
        }
        v20 = *(_DWORD *)(v11 + 88);
      }
      if ( v20 > 0 )
      {
        sub_2215DF0((__int64 *)&v49, *(_BYTE *)(v11 + 33));
        if ( v19 < 0 )
        {
          sub_2215CF0((__int64 *)&v49, -v19, *(_BYTE *)(v11 + 101));
          sub_2215BF0((__int64 *)&v49, v12, v16);
        }
        else
        {
          sub_2215BF0((__int64 *)&v49, (_BYTE *)v12 + v19, *(int *)(v11 + 88));
        }
      }
      v45 = *(_DWORD *)(a4 + 24) & 0xB0;
      v38 = *((_QWORD *)v49 - 3) + v39;
      if ( (*(_DWORD *)(a4 + 24) & 0x200) != 0 )
        v38 = *(_QWORD *)(v11 + 48) + *((_QWORD *)v49 - 3) + v39;
      v50[0] = (volatile signed __int32 *)&unk_4FD67D8;
      sub_2215AB0((__int64 *)v50, 2 * v38);
      v29 = a4;
      v27 = *(_QWORD *)(a4 + 16);
      v28 = (volatile signed __int32 **)&v48;
      LOBYTE(v29) = v27 > v38;
      v47 = v27 > v38 && v45 == 16;
      do
      {
        v30 = v29;
        v29 = *(unsigned __int8 *)v28;
        switch ( *(_BYTE *)v28 )
        {
          case 0:
            if ( v47 )
            {
              v31 = a5;
              goto LABEL_34;
            }
            break;
          case 1:
            v31 = a5;
            if ( v47 )
LABEL_34:
              v29 = (__int64)sub_2215CF0((__int64 *)v50, v27 - v38, v31);
            else
              v29 = sub_2215DF0((__int64 *)v50, a5);
            break;
          case 2:
            v29 = a4;
            if ( (*(_BYTE *)(a4 + 25) & 2) != 0 )
              v29 = (__int64)sub_2215BF0((__int64 *)v50, *(_BYTE **)(v11 + 40), *(_QWORD *)(v11 + 48));
            break;
          case 3:
            if ( v39 )
              v29 = sub_2215DF0((__int64 *)v50, *v43);
            break;
          case 4:
            v29 = (__int64)sub_2215B50((__int64 *)v50, &v49);
            break;
          default:
            v29 = v30;
            break;
        }
        v28 = (volatile signed __int32 **)((char *)v28 + 1);
      }
      while ( v28 != &v49 );
      if ( v39 > 1 )
      {
        sub_2215BF0((__int64 *)v50, v43 + 1, v39 - 1);
        v32 = v50[0];
        v33 = *((_QWORD *)v50[0] - 3);
        if ( v27 > v33 )
          goto LABEL_41;
      }
      else
      {
        v32 = v50[0];
        v33 = *((_QWORD *)v50[0] - 3);
        if ( v27 > v33 )
        {
LABEL_41:
          if ( v45 == 32 )
            sub_2215CF0((__int64 *)v50, v27 - v33, a5);
          else
            sub_22157B0(v50, 0, 0, v27 - v33, a5);
          v32 = v50[0];
LABEL_44:
          if ( !a3 )
            (*(__int64 (__fastcall **)(__int64, volatile signed __int32 *, _QWORD))(*(_QWORD *)a2 + 96LL))(
              a2,
              v32,
              (int)v27);
          sub_22159C0((__int64 *)v50);
          sub_22159C0((__int64 *)&v49);
          goto LABEL_10;
        }
      }
      LODWORD(v27) = v33;
      goto LABEL_44;
    }
  }
LABEL_10:
  *(_QWORD *)(a4 + 16) = 0;
  return a2;
}
