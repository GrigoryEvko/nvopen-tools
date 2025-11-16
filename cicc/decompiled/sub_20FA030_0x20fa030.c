// Function: sub_20FA030
// Address: 0x20fa030
//
__int64 __fastcall sub_20FA030(__int64 *a1)
{
  __int64 v2; // rax
  __int64 result; // rax
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // r12
  __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // r8
  unsigned int v25; // eax
  _QWORD *v26; // r15
  _QWORD *v27; // rdx
  __int64 v28; // r9
  __int64 v29; // rdi
  unsigned __int64 *v30; // r8
  __int64 v31; // rax
  __int64 v32; // r15
  __int64 v33; // rax
  _QWORD *v34; // r13
  __int64 v35; // r9
  __int64 v36; // rdi
  _QWORD *v37; // [rsp+8h] [rbp-58h]
  __int64 v38; // [rsp+10h] [rbp-50h]
  _QWORD *v39; // [rsp+10h] [rbp-50h]
  __int64 v40; // [rsp+10h] [rbp-50h]
  __int64 v41; // [rsp+18h] [rbp-48h]
  __int64 v42; // [rsp+18h] [rbp-48h]
  unsigned __int64 *v43; // [rsp+18h] [rbp-48h]
  _QWORD *v44; // [rsp+18h] [rbp-48h]
  __int64 v45[7]; // [rsp+28h] [rbp-38h] BYREF

  v2 = sub_160F9A0(a1[1], (__int64)&unk_4FC453D, 1u);
  if ( v2 )
  {
    result = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v2 + 104LL))(v2, &unk_4FC453D);
    if ( result )
      return result;
  }
  v4 = (__int64 *)a1[1];
  v5 = *v4;
  v6 = v4[1];
  if ( v5 == v6 )
LABEL_61:
    BUG();
  while ( *(_UNKNOWN **)v5 != &unk_4FC5828 )
  {
    v5 += 16;
    if ( v6 == v5 )
      goto LABEL_61;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(*(_QWORD *)(v5 + 8), &unk_4FC5828);
  v8 = sub_160F9A0(a1[1], (__int64)&unk_4FC6A0C, 1u);
  if ( !v8 )
  {
    v15 = sub_160F9A0(a1[1], (__int64)&unk_4FC62EC, 1u);
    if ( !v15 )
      goto LABEL_25;
    v11 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v15 + 104LL))(v15, &unk_4FC62EC);
LABEL_17:
    if ( v11 )
    {
LABEL_18:
      sub_20F9D50(v45);
      v16 = v45[0];
      v17 = a1[30];
      v45[0] = 0;
      a1[30] = v16;
      if ( v17 )
      {
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v17 + 8LL))(v17);
        if ( v45[0] )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v45[0] + 8LL))(v45[0]);
        v16 = a1[30];
      }
      v18 = v16 + 232;
      if ( !*(_QWORD *)(v11 + 1312) )
      {
        v23 = sub_22077B0(80);
        if ( v23 )
        {
          *(_QWORD *)(v23 + 24) = 0;
          *(_QWORD *)v23 = v23 + 16;
          *(_QWORD *)(v23 + 8) = 0x100000000LL;
          *(_QWORD *)(v23 + 32) = 0;
          *(_QWORD *)(v23 + 40) = 0;
          *(_DWORD *)(v23 + 48) = 0;
          *(_QWORD *)(v23 + 64) = 0;
          *(_BYTE *)(v23 + 72) = 0;
          *(_DWORD *)(v23 + 76) = 0;
        }
        v24 = *(_QWORD *)(v11 + 1312);
        *(_QWORD *)(v11 + 1312) = v23;
        if ( v24 )
        {
          v25 = *(_DWORD *)(v24 + 48);
          if ( v25 )
          {
            v26 = *(_QWORD **)(v24 + 32);
            v27 = &v26[2 * v25];
            do
            {
              if ( *v26 != -16 && *v26 != -8 )
              {
                v28 = v26[1];
                if ( v28 )
                {
                  v29 = *(_QWORD *)(v28 + 24);
                  if ( v29 )
                  {
                    v37 = v27;
                    v38 = v24;
                    v41 = v26[1];
                    j_j___libc_free_0(v29, *(_QWORD *)(v28 + 40) - v29);
                    v27 = v37;
                    v24 = v38;
                    v28 = v41;
                  }
                  v39 = v27;
                  v42 = v24;
                  j_j___libc_free_0(v28, 56);
                  v27 = v39;
                  v24 = v42;
                }
              }
              v26 += 2;
            }
            while ( v27 != v26 );
          }
          v43 = (unsigned __int64 *)v24;
          j___libc_free_0(*(_QWORD *)(v24 + 32));
          v30 = v43;
          if ( (unsigned __int64 *)*v43 != v43 + 2 )
          {
            _libc_free(*v43);
            v30 = v43;
          }
          j_j___libc_free_0(v30, 80);
        }
      }
      sub_1E06620(v11);
      sub_1E2B150(v18, *(_QWORD *)(v11 + 1312));
      v9 = a1[30];
      goto LABEL_10;
    }
LABEL_25:
    v19 = sub_22077B0(1320);
    v20 = v19;
    if ( v19 )
      sub_1E056B0(v19);
    v21 = a1[31];
    a1[31] = v20;
    if ( v21 )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v21 + 8LL))(v21);
      v20 = a1[31];
    }
    if ( !*(_QWORD *)(v20 + 1312) )
    {
      v31 = sub_22077B0(80);
      if ( v31 )
      {
        *(_QWORD *)(v31 + 24) = 0;
        *(_QWORD *)v31 = v31 + 16;
        *(_QWORD *)(v31 + 8) = 0x100000000LL;
        *(_QWORD *)(v31 + 32) = 0;
        *(_QWORD *)(v31 + 40) = 0;
        *(_DWORD *)(v31 + 48) = 0;
        *(_QWORD *)(v31 + 64) = 0;
        *(_BYTE *)(v31 + 72) = 0;
        *(_DWORD *)(v31 + 76) = 0;
      }
      v32 = *(_QWORD *)(v20 + 1312);
      *(_QWORD *)(v20 + 1312) = v31;
      if ( v32 )
      {
        v33 = *(unsigned int *)(v32 + 48);
        if ( (_DWORD)v33 )
        {
          v34 = *(_QWORD **)(v32 + 32);
          v44 = &v34[2 * v33];
          do
          {
            if ( *v34 != -8 && *v34 != -16 )
            {
              v35 = v34[1];
              if ( v35 )
              {
                v36 = *(_QWORD *)(v35 + 24);
                if ( v36 )
                {
                  v40 = v34[1];
                  j_j___libc_free_0(v36, *(_QWORD *)(v35 + 40) - v36);
                  v35 = v40;
                }
                j_j___libc_free_0(v35, 56);
              }
            }
            v34 += 2;
          }
          while ( v44 != v34 );
        }
        j___libc_free_0(*(_QWORD *)(v32 + 32));
        if ( *(_QWORD *)v32 != v32 + 16 )
          _libc_free(*(_QWORD *)v32);
        j_j___libc_free_0(v32, 80);
      }
    }
    sub_1E06620(v20);
    v22 = *(_QWORD *)(v20 + 1312);
    *(_QWORD *)(v22 + 64) = a1[32];
    sub_1E07D70(v22, 0);
    v11 = a1[31];
    goto LABEL_18;
  }
  v9 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v8 + 104LL))(v8, &unk_4FC6A0C);
  v10 = sub_160F9A0(a1[1], (__int64)&unk_4FC62EC, 1u);
  if ( !v10 )
  {
    if ( v9 )
      goto LABEL_10;
    goto LABEL_25;
  }
  v11 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v10 + 104LL))(v10, &unk_4FC62EC);
  if ( !v9 )
    goto LABEL_17;
LABEL_10:
  v12 = sub_22077B0(240);
  v13 = v12;
  if ( v12 )
    sub_1DDC180(v12);
  v14 = a1[29];
  a1[29] = v13;
  if ( v14 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v14 + 8LL))(v14);
    v13 = a1[29];
  }
  sub_1DE2CD0(v13, (_QWORD *)a1[32], v7, v9);
  return a1[29];
}
