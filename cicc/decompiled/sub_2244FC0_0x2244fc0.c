// Function: sub_2244FC0
// Address: 0x2244fc0
//
__int64 __fastcall sub_2244FC0(__int64 a1, __int64 a2, char a3, __int64 a4, wchar_t a5, const wchar_t **a6)
{
  _QWORD *v6; // r15
  __int64 v8; // rbp
  __int64 v9; // r12
  __int64 *v10; // rbx
  __int64 v11; // r14
  const wchar_t *v12; // r12
  __int64 v13; // rax
  int *v14; // rcx
  unsigned __int64 v15; // rdi
  size_t v16; // r15
  unsigned __int64 v18; // rsi
  __int64 v19; // rdx
  signed __int64 v20; // rbx
  int v21; // eax
  wchar_t *v22; // rdi
  _DWORD *v23; // rax
  const wchar_t *v24; // rdi
  _DWORD *v25; // rbp
  size_t v26; // rcx
  size_t v27; // rsi
  unsigned __int64 v28; // r15
  const wchar_t **v29; // rbx
  __int64 v30; // rax
  __int64 v31; // kr00_8
  const wchar_t *v32; // rsi
  unsigned __int64 v33; // rax
  const wchar_t *v34; // rdi
  int v35; // edx
  __int64 v36; // r14
  int v37; // eax
  unsigned __int64 v41; // [rsp+10h] [rbp-98h]
  int *v43; // [rsp+20h] [rbp-88h]
  __int64 v44; // [rsp+28h] [rbp-80h]
  unsigned __int64 v45; // [rsp+28h] [rbp-80h]
  char *v46; // [rsp+30h] [rbp-78h]
  int v47; // [rsp+30h] [rbp-78h]
  int v49; // [rsp+3Ch] [rbp-6Ch]
  bool v50; // [rsp+3Ch] [rbp-6Ch]
  int v51; // [rsp+5Ch] [rbp-4Ch] BYREF
  wchar_t *v52; // [rsp+60h] [rbp-48h] BYREF
  const wchar_t *v53[8]; // [rsp+68h] [rbp-40h] BYREF

  v6 = (_QWORD *)(a4 + 208);
  v8 = sub_2243120((_QWORD *)(a4 + 208), a2);
  v9 = sub_22091A0(&qword_4FD6A90);
  v10 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a4 + 208) + 24LL) + 8 * v9);
  v11 = *v10;
  if ( !*v10 )
  {
    v36 = sub_22077B0(0xA0u);
    *(_DWORD *)(v36 + 8) = 0;
    *(_QWORD *)(v36 + 16) = 0;
    *(_QWORD *)(v36 + 24) = 0;
    *(_QWORD *)v36 = off_4A048A0;
    *(_BYTE *)(v36 + 32) = 0;
    *(_QWORD *)(v36 + 36) = 0;
    *(_QWORD *)(v36 + 48) = 0;
    *(_QWORD *)(v36 + 56) = 0;
    *(_QWORD *)(v36 + 64) = 0;
    *(_QWORD *)(v36 + 72) = 0;
    *(_QWORD *)(v36 + 80) = 0;
    *(_QWORD *)(v36 + 88) = 0;
    *(_QWORD *)(v36 + 96) = 0;
    *(_DWORD *)(v36 + 104) = 0;
    *(_BYTE *)(v36 + 152) = 0;
    sub_2243C60(v36, v6);
    sub_2209690(*(_QWORD *)(a4 + 208), (volatile signed __int32 *)v36, v9);
    v11 = *v10;
  }
  v12 = *a6;
  v13 = *((_QWORD *)*a6 - 3);
  if ( **a6 == *(_DWORD *)(v11 + 108) )
  {
    v43 = *(int **)(v11 + 80);
    v18 = *(_QWORD *)(v11 + 88);
    v51 = *(_DWORD *)(v11 + 104);
    v41 = v18;
    if ( v13 )
      ++v12;
  }
  else
  {
    v14 = *(int **)(v11 + 64);
    v15 = *(_QWORD *)(v11 + 72);
    v51 = *(_DWORD *)(v11 + 100);
    v43 = v14;
    v41 = v15;
  }
  v16 = ((*(__int64 (__fastcall **)(__int64, __int64, const wchar_t *, const wchar_t *))(*(_QWORD *)v8 + 40LL))(
           v8,
           2048,
           v12,
           &v12[v13])
       - (__int64)v12) >> 2;
  if ( v16 )
  {
    v52 = (wchar_t *)&unk_4FD67F8;
    sub_2216730((__int64 *)&v52, 2 * v16);
    v19 = *(int *)(v11 + 96);
    v20 = v16 - v19;
    v21 = *(_DWORD *)(v11 + 96);
    if ( (__int64)(v16 - v19) > 0 )
    {
      if ( (int)v19 < 0 )
        v20 = v16;
      if ( *(_QWORD *)(v11 + 24) )
      {
        sub_2216430((const wchar_t **)&v52, 0, *((_QWORD *)v52 - 3), 2 * v20, 0);
        v22 = v52;
        v44 = *(_QWORD *)(v11 + 24);
        v46 = *(char **)(v11 + 16);
        v49 = *(_DWORD *)(v11 + 40);
        if ( *(v52 - 2) >= 0 )
        {
          sub_22163D0((const wchar_t **)&v52);
          v22 = v52;
        }
        v23 = sub_2244D30(v22, v49, v46, v44, (__int64)v12, (__int64)&v12[v20]);
        v24 = v52;
        v25 = v23;
        if ( *(v52 - 2) >= 0 )
        {
          sub_22163D0((const wchar_t **)&v52);
          v24 = v52;
        }
        v26 = *((_QWORD *)v24 - 3);
        v27 = v25 - v24;
        if ( v27 > v26 )
          sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", "basic_string::erase", v25 - v24, v26);
        sub_22161B0((const wchar_t **)&v52, v27, *((_QWORD *)v24 - 3) - v27, 0);
      }
      else
      {
        sub_2216530((const wchar_t **)&v52, v12, v20);
      }
      v21 = *(_DWORD *)(v11 + 96);
    }
    if ( v21 > 0 )
    {
      sub_2216A90((__int64 *)&v52, *(_DWORD *)(v11 + 36));
      if ( v20 < 0 )
      {
        sub_2216980((__int64 *)&v52, -v20, *(_DWORD *)(v11 + 112));
        sub_2216880((__int64 *)&v52, (unsigned __int64)v12, v16);
      }
      else
      {
        sub_2216880((__int64 *)&v52, (unsigned __int64)&v12[v20], *(int *)(v11 + 96));
      }
    }
    v45 = *((_QWORD *)v52 - 3) + v41;
    v47 = *(_DWORD *)(a4 + 24) & 0xB0;
    if ( (*(_DWORD *)(a4 + 24) & 0x200) != 0 )
      v45 = *(_QWORD *)(v11 + 56) + *((_QWORD *)v52 - 3) + v41;
    v53[0] = (const wchar_t *)&unk_4FD67F8;
    sub_2216730((__int64 *)v53, 2 * v45);
    v30 = a4;
    v28 = *(_QWORD *)(a4 + 16);
    v29 = (const wchar_t **)&v51;
    LOBYTE(v30) = v45 < v28;
    v50 = v45 < v28 && v47 == 16;
    do
    {
      v31 = v30;
      v30 = *(unsigned __int8 *)v29;
      switch ( *(_BYTE *)v29 )
      {
        case 0:
          if ( v50 )
            goto LABEL_31;
          break;
        case 1:
          if ( v50 )
LABEL_31:
            v30 = (__int64)sub_2216980((__int64 *)v53, v28 - v45, a5);
          else
            v30 = sub_2216A90((__int64 *)v53, a5);
          break;
        case 2:
          v30 = a4;
          if ( (*(_BYTE *)(a4 + 25) & 2) != 0 )
            v30 = (__int64)sub_2216880((__int64 *)v53, *(_QWORD *)(v11 + 48), *(_QWORD *)(v11 + 56));
          break;
        case 3:
          if ( v41 )
            v30 = sub_2216A90((__int64 *)v53, *v43);
          break;
        case 4:
          v30 = (__int64)sub_22167D0((__int64 *)v53, (const wchar_t **)&v52);
          break;
        default:
          v30 = v31;
          break;
      }
      v29 = (const wchar_t **)((char *)v29 + 1);
    }
    while ( v29 != (const wchar_t **)&v52 );
    if ( v41 > 1 )
    {
      sub_2216880((__int64 *)v53, (unsigned __int64)(v43 + 1), v41 - 1);
      v32 = v53[0];
      v33 = *((_QWORD *)v53[0] - 3);
      if ( v28 > v33 )
        goto LABEL_38;
    }
    else
    {
      v32 = v53[0];
      v33 = *((_QWORD *)v53[0] - 3);
      if ( v28 > v33 )
      {
LABEL_38:
        if ( v47 == 32 )
          sub_2216980((__int64 *)v53, v28 - v33, a5);
        else
          sub_2216430(v53, 0, 0, v28 - v33, a5);
        v32 = v53[0];
        goto LABEL_41;
      }
    }
    LODWORD(v28) = v33;
LABEL_41:
    if ( !a3 )
    {
      (*(void (__fastcall **)(__int64, const wchar_t *, _QWORD))(*(_QWORD *)a2 + 96LL))(a2, v32, (int)v28);
      v32 = v53[0];
    }
    if ( v32 - 6 != (const wchar_t *)&unk_4FD67E0 )
    {
      if ( &_pthread_key_create )
      {
        v37 = _InterlockedExchangeAdd((volatile signed __int32 *)v32 - 2, 0xFFFFFFFF);
      }
      else
      {
        v37 = *(v32 - 2);
        *((_DWORD *)v32 - 2) = v37 - 1;
      }
      if ( v37 <= 0 )
        j_j___libc_free_0_2((unsigned __int64)(v32 - 6));
    }
    v34 = v52 - 6;
    if ( v52 - 6 != (wchar_t *)&unk_4FD67E0 )
    {
      if ( &_pthread_key_create )
      {
        v35 = _InterlockedExchangeAdd(v52 - 2, 0xFFFFFFFF);
      }
      else
      {
        v35 = *(v52 - 2);
        *(v52 - 2) = v35 - 1;
      }
      if ( v35 <= 0 )
        j_j___libc_free_0_2((unsigned __int64)v34);
    }
  }
  *(_QWORD *)(a4 + 16) = 0;
  return a2;
}
