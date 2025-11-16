// Function: sub_C54750
// Address: 0xc54750
//
_BYTE *__fastcall sub_C54750(__int128 a1, unsigned int a2)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  _BYTE *result; // rax
  unsigned int v6; // r13d
  __int64 v7; // rsi
  __int128 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r12
  __int64 v12; // rax
  _DWORD *v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  _BYTE *v18; // rax
  unsigned __int8 v19; // al
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  int v26; // ecx
  unsigned int v27; // r15d
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdi
  int v31; // ecx
  __int128 v32; // rdi
  __int64 v33; // rdi
  size_t v34; // rdx
  size_t v35; // r12
  unsigned __int8 v36; // al
  __int64 v37; // rdx
  int v38; // r12d
  unsigned int v39; // r15d
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rdi
  __int64 v44; // rax
  __int64 v45; // rax
  int v46; // ecx
  __int64 v47; // rdi
  int v48; // [rsp+4h] [rbp-6Ch]
  __int64 v49; // [rsp+8h] [rbp-68h]
  int v50; // [rsp+10h] [rbp-60h]
  const void *v51; // [rsp+10h] [rbp-60h]
  __int64 v52; // [rsp+18h] [rbp-58h]
  __int64 v53; // [rsp+18h] [rbp-58h]
  __int64 v54; // [rsp+20h] [rbp-50h] BYREF
  __int64 v55; // [rsp+28h] [rbp-48h]
  __int64 v56; // [rsp+30h] [rbp-40h]

  v3 = *((_QWORD *)&a1 + 1);
  v4 = a1;
  if ( *(_QWORD *)(*((_QWORD *)&a1 + 1) + 32LL) )
  {
    v19 = *(_BYTE *)(*((_QWORD *)&a1 + 1) + 12LL);
    if ( (v19 & 0x18) != 0 )
    {
      if ( ((v19 >> 3) & 3) != 1 )
        goto LABEL_14;
    }
    else
    {
      *(_QWORD *)&a1 = *((_QWORD *)&a1 + 1);
      if ( (*(unsigned int (__fastcall **)(_QWORD))(**((_QWORD **)&a1 + 1) + 8LL))(*((_QWORD *)&a1 + 1)) != 1 )
      {
LABEL_14:
        v20 = sub_CB7210(a1, *((_QWORD *)&a1 + 1));
        v21 = *(_QWORD *)(v3 + 24);
        v56 = 2;
        v22 = v20;
        v23 = *(_QWORD *)(v3 + 32);
        v54 = v21;
        v55 = v23;
        v24 = sub_C51AE0(v22, (__int64)&v54);
        sub_A51340(v24, "=<value>", qword_4979A98);
        v25 = *(_QWORD *)(v3 + 32);
        if ( v25 == 1 )
          v26 = qword_4C5C728 + 6;
        else
          v26 = v25 + qword_4C5C718 + 5;
        sub_C540D0(*(_OWORD *)(v3 + 40), a2, v26 + 8);
        result = (_BYTE *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 16LL))(v4);
        v48 = (int)result;
        if ( !(_DWORD)result )
          return result;
        v27 = 0;
        while ( 1 )
        {
          v33 = v4;
          v51 = (const void *)(*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v4 + 24LL))(v4, v27);
          v35 = v34;
          v49 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v4 + 32LL))(v4, v27);
          v36 = *(_BYTE *)(v3 + 12);
          v53 = v37;
          if ( (v36 & 0x18) != 0 )
          {
            if ( ((v36 >> 3) & 3) != 1 )
              goto LABEL_19;
          }
          else
          {
            v33 = v3;
            if ( (*(unsigned int (__fastcall **)(__int64))(*(_QWORD *)v3 + 8LL))(v3) != 1 )
              goto LABEL_19;
          }
          result = (_BYTE *)(v35 | v53);
          if ( v35 | v53 )
          {
LABEL_19:
            v28 = sub_CB7210(v33, v27);
            v29 = *(_QWORD *)(v28 + 32);
            v30 = v28;
            if ( (unsigned __int64)(*(_QWORD *)(v28 + 24) - v29) <= 4 )
            {
              v30 = sub_CB6200(v28, "    =", 5);
            }
            else
            {
              *(_DWORD *)v29 = 538976288;
              *(_BYTE *)(v29 + 4) = 61;
              *(_QWORD *)(v28 + 32) += 5LL;
            }
            sub_A51340(v30, v51, v35);
            v31 = v35 + 8;
            if ( !v35 )
            {
              v30 = sub_CB7210(v30, v51);
              sub_A51340(v30, "<empty>", qword_4979A88);
              v31 = 15;
            }
            *((_QWORD *)&v32 + 1) = v53;
            if ( v53 )
            {
              *(_QWORD *)&v32 = v49;
              result = (_BYTE *)sub_C54430(v32, a2, v31);
            }
            else
            {
              v47 = sub_CB7210(v30, 0);
              result = *(_BYTE **)(v47 + 32);
              if ( (unsigned __int64)result >= *(_QWORD *)(v47 + 24) )
              {
                result = (_BYTE *)sub_CB5D20(v47, 10);
              }
              else
              {
                *(_QWORD *)(v47 + 32) = result + 1;
                *result = 10;
              }
            }
          }
          if ( v48 == ++v27 )
            return result;
        }
      }
    }
    *(_QWORD *)&a1 = v4;
    v38 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 16LL))(v4);
    if ( v38 )
    {
      v39 = 0;
      while ( 1 )
      {
        *((_QWORD *)&a1 + 1) = v39;
        *(_QWORD *)&a1 = v4;
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v4 + 24LL))(v4, v39);
        if ( !v40 )
          break;
        if ( v38 == ++v39 )
          goto LABEL_14;
      }
      v41 = sub_CB7210(v4, v39);
      v42 = *(_QWORD *)(v3 + 24);
      v56 = 2;
      v43 = v41;
      v44 = *(_QWORD *)(v3 + 32);
      v54 = v42;
      v55 = v44;
      sub_C51AE0(v43, (__int64)&v54);
      v45 = *(_QWORD *)(v3 + 32);
      if ( v45 == 1 )
        v46 = qword_4C5C728 + 6;
      else
        v46 = v45 + qword_4C5C718 + 5;
      a1 = *(_OWORD *)(v3 + 40);
      sub_C540D0(a1, a2, v46);
    }
    goto LABEL_14;
  }
  if ( *(_QWORD *)(*((_QWORD *)&a1 + 1) + 48LL) )
  {
    v15 = sub_CB7210(a1, *((_QWORD *)&a1 + 1));
    v16 = sub_904010(v15, "  ");
    v17 = sub_A51340(v16, *(const void **)(*((_QWORD *)&a1 + 1) + 40LL), *(_QWORD *)(*((_QWORD *)&a1 + 1) + 48LL));
    v18 = *(_BYTE **)(v17 + 32);
    if ( (unsigned __int64)v18 >= *(_QWORD *)(v17 + 24) )
    {
      sub_CB5D20(v17, 10);
    }
    else
    {
      *(_QWORD *)(v17 + 32) = v18 + 1;
      *v18 = 10;
    }
  }
  result = (_BYTE *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 16LL))(v4);
  v50 = (int)result;
  if ( (_DWORD)result )
  {
    v6 = 0;
    do
    {
      v9 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v4 + 24LL))(v4, v6);
      v11 = v10;
      v52 = v9;
      v12 = sub_CB7210(v4, v6);
      v13 = *(_DWORD **)(v12 + 32);
      v14 = v12;
      if ( *(_QWORD *)(v12 + 24) - (_QWORD)v13 > 3u )
      {
        *v13 = 538976288;
        *(_QWORD *)(v12 + 32) += 4LL;
      }
      else
      {
        v14 = sub_CB6200(v12, "    ", 4);
      }
      v55 = v11;
      v56 = 2;
      v54 = v52;
      sub_C51AE0(v14, (__int64)&v54);
      v7 = v6++;
      *(_QWORD *)&v8 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v4 + 32LL))(v4, v7);
      result = (_BYTE *)sub_C540D0(v8, a2, (int)v11 + 8);
    }
    while ( v50 != v6 );
  }
  return result;
}
