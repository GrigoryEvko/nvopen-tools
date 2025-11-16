// Function: sub_34B3480
// Address: 0x34b3480
//
void __fastcall sub_34B3480(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned int v3; // r12d
  __int64 v4; // rax
  __int64 v5; // r15
  __int64 v6; // r14
  unsigned __int64 v7; // rdi
  __int64 v8; // rdx
  int v9; // ecx
  bool v10; // al
  __int64 *v11; // rax
  unsigned int *v12; // r13
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // r15
  char *v16; // r8
  __int64 v17; // rdx
  __int64 v18; // r12
  __int64 v19; // r13
  char *v20; // r14
  unsigned __int16 v21; // bx
  __int64 v22; // rax
  int v23; // edx
  __int64 v24; // rax
  unsigned __int16 *v25; // rax
  unsigned __int16 v26; // cx
  __int64 v27; // rax
  __int64 v28; // r14
  __int64 v29; // r15
  char *v30; // rax
  __int64 v31; // rdx
  char *i; // r12
  __int64 v33; // rax
  int v34; // edx
  __int64 *v35; // [rsp+8h] [rbp-B8h]
  bool v36; // [rsp+17h] [rbp-A9h]
  __int64 *v37; // [rsp+18h] [rbp-A8h]
  unsigned int *v38; // [rsp+20h] [rbp-A0h]
  unsigned __int16 *v39; // [rsp+20h] [rbp-A0h]
  unsigned int *v40; // [rsp+28h] [rbp-98h]
  unsigned __int16 *v41; // [rsp+28h] [rbp-98h]
  unsigned __int16 *v42; // [rsp+38h] [rbp-88h]
  unsigned __int16 v43; // [rsp+38h] [rbp-88h]
  char *v44; // [rsp+40h] [rbp-80h] BYREF
  char v45; // [rsp+50h] [rbp-70h] BYREF

  v2 = a1;
  v3 = *(_DWORD *)(*(_QWORD *)(a1 + 32) + 16LL);
  v4 = sub_22077B0(0x98u);
  v5 = v4;
  if ( v4 )
    sub_34B31E0(v4, v3, a2);
  *(_QWORD *)(a1 + 120) = v5;
  v6 = a2 + 48;
  v7 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a2 + 48 == v7 )
  {
    v36 = 0;
    goto LABEL_13;
  }
  if ( !v7 )
    BUG();
  v8 = *(_QWORD *)v7;
  v9 = *(_DWORD *)(v7 + 44);
  if ( (*(_QWORD *)v7 & 4) == 0 )
  {
    if ( (v9 & 4) != 0 )
    {
      while ( 1 )
      {
        v7 = v8 & 0xFFFFFFFFFFFFFFF8LL;
        v9 = *(_DWORD *)((v8 & 0xFFFFFFFFFFFFFFF8LL) + 44) & 0xFFFFFF;
        if ( (*(_DWORD *)((v8 & 0xFFFFFFFFFFFFFFF8LL) + 44) & 4) == 0 )
          break;
        v8 = *(_QWORD *)v7;
      }
    }
LABEL_10:
    if ( (v9 & 8) != 0 )
    {
      v10 = sub_2E88A90(v7, 32, 1);
      v5 = *(_QWORD *)(v2 + 120);
      v36 = v10;
      goto LABEL_13;
    }
    goto LABEL_43;
  }
  if ( (v9 & 4) == 0 )
    goto LABEL_10;
LABEL_43:
  v36 = (*(_QWORD *)(*(_QWORD *)(v7 + 16) + 24LL) & 0x20LL) != 0;
LABEL_13:
  v11 = *(__int64 **)(a2 + 112);
  v37 = v11;
  v35 = &v11[*(unsigned int *)(a2 + 120)];
  if ( v35 != v11 )
  {
    do
    {
      v12 = *(unsigned int **)(*v37 + 192);
      v38 = v12;
      v40 = (unsigned int *)sub_2E33140(*v37);
      if ( v12 != v40 )
      {
        v13 = v6;
        v14 = v5;
        v15 = v13;
        do
        {
          v16 = sub_E922F0(*(_QWORD **)(v2 + 32), *v40);
          v42 = (unsigned __int16 *)&v16[2 * v17];
          if ( v16 != (char *)v42 )
          {
            v18 = v14;
            v19 = v2;
            v20 = v16;
            do
            {
              v21 = *(_WORD *)v20;
              sub_34B3410(*(_QWORD **)(v19 + 120), *(unsigned __int16 *)v20, 0);
              v22 = *(_QWORD *)(a2 + 56);
              if ( v22 == v15 )
              {
                v23 = 0;
              }
              else
              {
                v23 = 0;
                do
                {
                  v22 = *(_QWORD *)(v22 + 8);
                  ++v23;
                }
                while ( v22 != v15 );
              }
              v20 += 2;
              *(_DWORD *)(*(_QWORD *)(v18 + 104) + 4LL * v21) = v23;
              *(_DWORD *)(*(_QWORD *)(v18 + 128) + 4LL * v21) = -1;
            }
            while ( v42 != (unsigned __int16 *)v20 );
            v14 = v18;
            v2 = v19;
          }
          v40 += 6;
        }
        while ( v38 != v40 );
        v24 = v15;
        v5 = v14;
        v6 = v24;
      }
      ++v37;
    }
    while ( v35 != v37 );
  }
  sub_2E76F80((__int64)&v44, *(_QWORD *)(*(_QWORD *)(v2 + 8) + 48LL), *(_QWORD *)(v2 + 8));
  v25 = sub_2EBFBC0(*(_QWORD **)(*(_QWORD *)(v2 + 8) + 32LL));
  v26 = *v25;
  v39 = v25;
  if ( *v25 )
  {
    v27 = v6;
    v28 = v5;
    v29 = v27;
    do
    {
      if ( v36 || (*(_QWORD *)&v44[8 * (v26 >> 6)] & (1LL << v26)) != 0 )
      {
        v30 = sub_E922F0(*(_QWORD **)(v2 + 32), v26);
        v41 = (unsigned __int16 *)&v30[2 * v31];
        for ( i = v30; v41 != (unsigned __int16 *)i; *(_DWORD *)(*(_QWORD *)(v28 + 128) + 4LL * v43) = -1 )
        {
          v43 = *(_WORD *)i;
          sub_34B3410(*(_QWORD **)(v2 + 120), *(unsigned __int16 *)i, 0);
          v33 = *(_QWORD *)(a2 + 56);
          if ( v33 == v29 )
          {
            v34 = 0;
          }
          else
          {
            v34 = 0;
            do
            {
              v33 = *(_QWORD *)(v33 + 8);
              ++v34;
            }
            while ( v33 != v29 );
          }
          i += 2;
          *(_DWORD *)(*(_QWORD *)(v28 + 104) + 4LL * v43) = v34;
        }
      }
      v26 = *++v39;
    }
    while ( *v39 );
  }
  if ( v44 != &v45 )
    _libc_free((unsigned __int64)v44);
}
