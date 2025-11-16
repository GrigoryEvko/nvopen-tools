// Function: sub_16B2860
// Address: 0x16b2860
//
__int64 __fastcall sub_16B2860(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v5; // r13d
  __int64 result; // rax
  unsigned int v7; // r15d
  void *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rsi
  __int128 v11; // rax
  __int64 v12; // rax
  size_t v13; // rdx
  size_t v14; // rbx
  const void *v15; // r12
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdi
  _BYTE *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned int v26; // ebx
  const char *v27; // rax
  size_t v28; // rdx
  __int64 v29; // rcx
  void *v30; // rdi
  const char *v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r12
  __int64 v36; // rax
  size_t v37; // rdx
  _BYTE *v38; // rdi
  const char *v39; // rsi
  unsigned __int64 v40; // rax
  __int64 v41; // rdx
  int v42; // r12d
  __int64 v43; // rcx
  __int64 v44; // rax
  unsigned int v45; // r12d
  __int64 v46; // rdx
  __int64 v47; // r15
  __int64 v48; // rax
  __int64 v49; // [rsp+0h] [rbp-40h]
  size_t v50; // [rsp+0h] [rbp-40h]
  size_t v51; // [rsp+0h] [rbp-40h]
  int v52; // [rsp+Ch] [rbp-34h]
  int v53; // [rsp+Ch] [rbp-34h]

  v5 = a3;
  if ( *(_QWORD *)(a2 + 32) )
  {
    v24 = sub_16E8C20(a1, a2, a3, a4);
    v25 = sub_1263B40(v24, "  -");
    sub_1549FF0(v25, *(const char **)(a2 + 24), *(_QWORD *)(a2 + 32));
    v26 = 0;
    sub_16B2520(*(_OWORD *)(a2 + 40), v5, *(_QWORD *)(a2 + 32) + 6);
    result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v53 = result;
    if ( (_DWORD)result )
    {
      do
      {
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 24LL))(a1, v26);
        v42 = -8 - v41;
        v44 = sub_16E8C20(a1, v26, v41, v43);
        v45 = v5 + v42;
        v46 = *(_QWORD *)(v44 + 24);
        v47 = v44;
        if ( (unsigned __int64)(*(_QWORD *)(v44 + 16) - v46) > 4 )
        {
          *(_DWORD *)v46 = 538976288;
          *(_BYTE *)(v46 + 4) = 61;
          *(_QWORD *)(v44 + 24) += 5LL;
        }
        else
        {
          v47 = sub_16E7EE0(v44, "    =", 5);
        }
        v27 = (const char *)(*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 24LL))(a1, v26);
        v30 = *(void **)(v47 + 24);
        v31 = v27;
        if ( *(_QWORD *)(v47 + 16) - (_QWORD)v30 < v28 )
        {
          v30 = (void *)v47;
          sub_16E7EE0(v47, v27);
        }
        else if ( v28 )
        {
          v51 = v28;
          memcpy(v30, v27, v28);
          v28 = v51;
          *(_QWORD *)(v47 + 24) += v51;
        }
        v32 = sub_16E8C20(v30, v31, v28, v29);
        v33 = sub_16E8750(v32, v45);
        v34 = *(_QWORD *)(v33 + 24);
        v35 = v33;
        if ( (unsigned __int64)(*(_QWORD *)(v33 + 16) - v34) <= 4 )
        {
          v35 = sub_16E7EE0(v33, " -   ", 5);
        }
        else
        {
          *(_DWORD *)v34 = 538979616;
          *(_BYTE *)(v34 + 4) = 32;
          *(_QWORD *)(v33 + 24) += 5LL;
        }
        v36 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 32LL))(a1, v26);
        v38 = *(_BYTE **)(v35 + 24);
        v39 = (const char *)v36;
        v40 = *(_QWORD *)(v35 + 16);
        if ( v40 - (unsigned __int64)v38 < v37 )
        {
          v48 = sub_16E7EE0(v35, v39);
          v38 = *(_BYTE **)(v48 + 24);
          v35 = v48;
          v40 = *(_QWORD *)(v48 + 16);
        }
        else if ( v37 )
        {
          v50 = v37;
          memcpy(v38, v39, v37);
          v40 = *(_QWORD *)(v35 + 16);
          v38 = (_BYTE *)(v50 + *(_QWORD *)(v35 + 24));
          *(_QWORD *)(v35 + 24) = v38;
        }
        if ( v40 <= (unsigned __int64)v38 )
        {
          result = sub_16E7DE0(v35, 10);
        }
        else
        {
          result = (__int64)(v38 + 1);
          *(_QWORD *)(v35 + 24) = v38 + 1;
          *v38 = 10;
        }
        ++v26;
      }
      while ( v53 != v26 );
    }
  }
  else
  {
    if ( *(_QWORD *)(a2 + 48) )
    {
      v20 = sub_16E8C20(a1, a2, a3, a4);
      v21 = sub_1263B40(v20, "  ");
      v22 = sub_1549FF0(v21, *(const char **)(a2 + 40), *(_QWORD *)(a2 + 48));
      v23 = *(_BYTE **)(v22 + 24);
      if ( (unsigned __int64)v23 >= *(_QWORD *)(v22 + 16) )
      {
        sub_16E7DE0(v22, 10);
      }
      else
      {
        *(_QWORD *)(v22 + 24) = v23 + 1;
        *v23 = 10;
      }
    }
    result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v52 = result;
    if ( (_DWORD)result )
    {
      v7 = 0;
      do
      {
        v12 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 24LL))(a1, v7);
        v14 = v13;
        v15 = (const void *)v12;
        v17 = sub_16E8C20(a1, v7, v13, v16);
        v18 = *(_QWORD *)(v17 + 24);
        v19 = v17;
        if ( (unsigned __int64)(*(_QWORD *)(v17 + 16) - v18) > 4 )
        {
          *(_DWORD *)v18 = 538976288;
          *(_BYTE *)(v18 + 4) = 45;
          v8 = (void *)(*(_QWORD *)(v17 + 24) + 5LL);
          v9 = *(_QWORD *)(v17 + 16);
          *(_QWORD *)(v19 + 24) = v8;
          if ( v14 <= v9 - (__int64)v8 )
            goto LABEL_6;
        }
        else
        {
          v19 = sub_16E7EE0(v17, "    -", 5);
          v8 = *(void **)(v19 + 24);
          if ( v14 <= *(_QWORD *)(v19 + 16) - (_QWORD)v8 )
          {
LABEL_6:
            if ( v14 )
            {
              v49 = v19;
              memcpy(v8, v15, v14);
              *(_QWORD *)(v49 + 24) += v14;
            }
            goto LABEL_8;
          }
        }
        sub_16E7EE0(v19, (const char *)v15, v14);
LABEL_8:
        v10 = v7++;
        *(_QWORD *)&v11 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 32LL))(a1, v10);
        result = sub_16B2520(v11, v5, (int)v14 + 8);
      }
      while ( v52 != v7 );
    }
  }
  return result;
}
