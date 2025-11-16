// Function: sub_31EBBA0
// Address: 0x31ebba0
//
__int64 __fastcall sub_31EBBA0(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4, unsigned __int8 a5)
{
  unsigned __int8 v6; // al
  __int64 result; // rax
  __int64 v8; // rdx
  _BYTE *v9; // rdi
  char *v10; // rax
  size_t v11; // rdx
  void *v12; // r8
  size_t v13; // r12
  _BYTE *v14; // rax
  size_t v15; // r14
  size_t v16; // rdx
  char *v17; // r12
  size_t v18; // rdx
  char *v19; // rax
  __int64 v20; // r12
  int v21; // eax
  __int64 *v22; // r12
  __int64 v23; // rdx
  __int64 *v24; // rax
  __int64 v25; // r13
  __int64 *v26; // rbx
  __int64 *v27; // r14
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rdi
  __int64 v31; // rdi
  __int64 v32; // rax
  unsigned __int64 v33; // rax
  unsigned __int8 v34; // al
  __int64 v35; // r14
  __int64 v36; // rdx
  char *v37; // r13
  __int64 *v38; // rdi
  _QWORD *v39; // rsi
  __int64 (__fastcall *v40)(__int64, unsigned __int64, _BYTE *, unsigned int); // rax
  __int64 v41; // r12
  __int64 v42; // r14
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rax
  _BYTE **v48; // [rsp+10h] [rbp-90h]
  bool v50; // [rsp+1Fh] [rbp-81h]
  __int64 v51; // [rsp+20h] [rbp-80h]
  _BYTE **v52; // [rsp+28h] [rbp-78h]
  __int64 v53; // [rsp+30h] [rbp-70h]
  char *s1; // [rsp+38h] [rbp-68h]
  void *s1b; // [rsp+38h] [rbp-68h]
  void *s1c; // [rsp+38h] [rbp-68h]
  char *s1a; // [rsp+38h] [rbp-68h]
  __int64 v58[4]; // [rsp+40h] [rbp-60h] BYREF
  char v59; // [rsp+60h] [rbp-40h]
  char v60; // [rsp+61h] [rbp-3Fh]

  v6 = *(_BYTE *)(a2 - 16);
  if ( (v6 & 2) != 0 )
  {
    result = *(_QWORD *)(a2 - 32);
    v8 = *(unsigned int *)(a2 - 24);
  }
  else
  {
    v8 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
    result = a2 - 8LL * ((v6 >> 2) & 0xF) - 16;
  }
  v52 = (_BYTE **)result;
  v48 = (_BYTE **)(result + 8 * v8);
  v50 = 0;
  if ( v48 != (_BYTE **)result )
  {
    do
    {
      v9 = *v52;
      if ( **v52 )
      {
        v34 = *(v9 - 16);
        if ( (v34 & 2) != 0 )
        {
          v35 = *((_QWORD *)v9 - 4);
          v36 = *((unsigned int *)v9 - 6);
        }
        else
        {
          v36 = (*((_WORD *)v9 - 8) >> 6) & 0xF;
          v35 = (__int64)&v9[-8 * ((v34 >> 2) & 0xF) - 16];
        }
        s1a = (char *)(v35 + 8 * v36);
        if ( s1a != (char *)v35 )
        {
          v37 = (char *)v35;
          do
          {
            v41 = *(_QWORD *)(*(_QWORD *)v37 + 136LL);
            v42 = sub_B2BEC0(a1[4]);
            v43 = sub_9208B0(v42, *(_QWORD *)(v41 + 8));
            v58[1] = v44;
            v58[0] = (unsigned __int64)(v43 + 7) >> 3;
            v45 = sub_CA1930(v58);
            if ( *(_BYTE *)v41 == 17 && v50 && (unsigned __int64)(v45 - 2) <= 6 )
            {
              v38 = (__int64 *)a1[2];
              v39 = *(_QWORD **)(v41 + 24);
              v40 = *(__int64 (__fastcall **)(__int64, unsigned __int64, _BYTE *, unsigned int))(*v38 + 424);
              if ( *(_DWORD *)(v41 + 32) > 0x40u )
                v39 = (_QWORD *)*v39;
              if ( v40 == sub_31D54C0 )
                sub_E98EB0(v38[28], (unsigned __int64)v39, 0);
              else
                v40((__int64)v38, (unsigned __int64)v39, 0, 0);
            }
            else
            {
              sub_31EA6F0(a1[2], v42, v41, 0);
            }
            v37 += 8;
          }
          while ( s1a != v37 );
        }
      }
      else
      {
        v10 = (char *)sub_B91420((__int64)v9);
        v12 = v10;
        v13 = v11;
        if ( !v11 )
          goto LABEL_42;
        s1 = v10;
        v14 = memchr(v10, 33, v11);
        v12 = s1;
        v15 = v14 - s1;
        if ( !v14 )
          v15 = -1;
        if ( v13 >= v15 )
        {
          v16 = v13;
          v17 = &s1[v15];
          v18 = v16 - v15;
          if ( v18 && (v19 = (char *)memchr(&s1[v15], 67, v18), v12 = s1, v19) )
          {
            v53 = v19 - v17;
            v50 = v19 - v17 != -1;
          }
          else
          {
            v50 = 0;
            v53 = -1;
          }
        }
        else
        {
LABEL_42:
          v50 = 0;
          v15 = v13;
          v53 = -1;
        }
        v20 = *a1;
        if ( *(_QWORD *)(*a1 + 8) != v15
          || v15 && (s1b = v12, v21 = memcmp(v12, *(const void **)v20, v15), v12 = s1b, v21) )
        {
          s1c = v12;
          v32 = sub_31DA6B0(*(_QWORD *)(v20 + 16));
          v33 = sub_E8A090(v32, (size_t)s1c, v15, *(_QWORD *)(*(_QWORD *)(v20 + 24) + 72LL));
          (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(v20 + 16) + 224LL) + 176LL))(
            *(_QWORD *)(*(_QWORD *)(v20 + 16) + 224LL),
            v33,
            0);
          v12 = s1c;
          *(_QWORD *)(v20 + 8) = v15;
          *(_QWORD *)v20 = s1c;
        }
        v22 = &a3[a4];
        v23 = *a3;
        if ( a3 != v22 )
        {
          v24 = a1;
          v25 = *a3;
          v26 = a3 + 1;
          v27 = v24;
          v28 = a5 ^ 1u;
          while ( 2 )
          {
            if ( v25 == v23 || a5 != 1 )
            {
              v31 = *(_QWORD *)(v27[1] + 24);
              v60 = 1;
              v58[0] = (__int64)"pcsection_base";
              v59 = 3;
              v51 = sub_E6C380(v31, v58, 1, v28, (__int64)v12);
              (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(v27[2] + 224) + 208LL))(
                *(_QWORD *)(v27[2] + 224),
                v51,
                0);
              sub_31DCA50(v27[2]);
            }
            else
            {
              v30 = v27[2];
              if ( v53 == -1 )
              {
                sub_31DCA50(v30);
                if ( v22 == v26 )
                {
LABEL_24:
                  a1 = v27;
                  break;
                }
                goto LABEL_19;
              }
              sub_31DCA60(v30);
            }
            if ( v22 == v26 )
              goto LABEL_24;
LABEL_19:
            v29 = *v26;
            v23 = v25;
            ++v26;
            v25 = v29;
            continue;
          }
        }
      }
      result = (__int64)++v52;
    }
    while ( v48 != v52 );
  }
  return result;
}
