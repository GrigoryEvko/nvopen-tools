// Function: sub_29F51A0
// Address: 0x29f51a0
//
char __fastcall sub_29F51A0(unsigned __int8 *a1, unsigned __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 **v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // r15
  unsigned __int8 *v17; // r14
  unsigned __int8 *v18; // r15
  _BYTE *v19; // rdi
  int v20; // edx
  unsigned __int8 **v21; // rax
  char *v22; // r14
  char *v23; // rsi
  __int64 v24; // r13
  __int64 v25; // rcx
  unsigned __int64 v26; // r8
  unsigned __int64 v27; // rax
  unsigned __int8 **v28; // rax
  __int64 *v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // r15
  char *v32; // r14
  unsigned __int64 v33; // rax
  _QWORD *v34; // r15
  size_t v35; // rdx
  unsigned __int64 v36; // rax
  unsigned __int64 v37; // rax
  __int64 v38; // rdx
  unsigned __int64 v39; // r14
  __int64 v40; // rax
  const void *v41; // rsi
  __int64 v42; // rax
  unsigned __int64 v44; // [rsp+8h] [rbp-38h]

  if ( *(_BYTE *)(a3 + 28) )
  {
    v9 = *(unsigned __int8 ***)(a3 + 8);
    v10 = (__int64)&v9[*(unsigned int *)(a3 + 20)];
    LODWORD(v11) = *(_DWORD *)(a3 + 20);
    v12 = (__int64)v9;
    if ( v9 != (unsigned __int8 **)v10 )
    {
      while ( *(unsigned __int8 **)v12 != a1 )
      {
        v12 += 8;
        if ( v10 == v12 )
          goto LABEL_8;
      }
      return v12;
    }
    goto LABEL_38;
  }
  v12 = (__int64)sub_C8CA60(a3, (__int64)a1);
  if ( v12 )
    return v12;
  if ( !*(_BYTE *)(a3 + 28) )
  {
LABEL_35:
    sub_C8CC70(a3, (__int64)a1, (__int64)v9, v10, a5, a6);
    goto LABEL_9;
  }
  v9 = *(unsigned __int8 ***)(a3 + 8);
  v11 = *(unsigned int *)(a3 + 20);
  v12 = (__int64)&v9[v11];
  if ( v9 == (unsigned __int8 **)v12 )
  {
LABEL_38:
    if ( *(_DWORD *)(a3 + 16) > (unsigned int)v11 )
    {
      *(_DWORD *)(a3 + 20) = v11 + 1;
      *(_QWORD *)v12 = a1;
      ++*(_QWORD *)a3;
      goto LABEL_9;
    }
    goto LABEL_35;
  }
LABEL_8:
  while ( *v9 != a1 )
  {
    if ( ++v9 == (unsigned __int8 **)v12 )
      goto LABEL_38;
  }
LABEL_9:
  v13 = *((_QWORD *)a1 + 5);
  v12 = sub_AA5BA0(v13);
  v14 = v12;
  if ( v12 != v13 + 48 )
  {
    v15 = v12 - 24;
    if ( v14 )
      v14 = v15;
    LOBYTE(v12) = sub_B445A0((__int64)a1, v14);
    if ( !(_BYTE)v12 )
    {
      v16 = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
      if ( (a1[7] & 0x40) != 0 )
      {
        v17 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
        v18 = &v17[v16];
      }
      else
      {
        v17 = &a1[-v16];
        v18 = a1;
      }
      for ( ; v18 != v17; v17 += 32 )
      {
        v19 = *(_BYTE **)v17;
        if ( **(_BYTE **)v17 > 0x1Cu && *((_QWORD *)v19 + 5) == *((_QWORD *)a1 + 5) )
          sub_29F51A0(v19, a2, a3);
      }
      v20 = *a1;
      LOBYTE(v12) = v20 - 30;
      if ( (unsigned int)(v20 - 30) > 0xA )
      {
        if ( (_BYTE)v20 == 85 )
        {
          LOWORD(v12) = *((_WORD *)a1 + 1) & 3;
          if ( (_WORD)v12 == 2 )
            return v12;
          LODWORD(v12) = sub_B49240((__int64)a1);
          if ( (_DWORD)v12 == 146 )
            return v12;
          LODWORD(v12) = sub_B49240((__int64)a1);
          if ( (_DWORD)v12 == 143 )
            return v12;
          LODWORD(v12) = sub_B49240((__int64)a1);
          if ( (_DWORD)v12 == 144 )
            return v12;
          LOBYTE(v20) = *a1;
        }
        if ( (_BYTE)v20 != 78
          || (v12 = *((_QWORD *)a1 - 4), *(_BYTE *)v12 != 85)
          || (LOWORD(v12) = *(_WORD *)(v12 + 2) & 3, (_WORD)v12 != 2) )
        {
          v21 = (unsigned __int8 **)a2[6];
          if ( v21 == (unsigned __int8 **)(a2[8] - 8) )
          {
            v22 = (char *)a2[9];
            v23 = (char *)a2[5];
            v24 = v22 - v23;
            v25 = (v22 - v23) >> 3;
            if ( ((__int64)(a2[4] - a2[2]) >> 3) + ((v25 - 1) << 6) + ((__int64)((__int64)v21 - a2[7]) >> 3) == 0xFFFFFFFFFFFFFFFLL )
              sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
            v26 = *a2;
            v27 = a2[1];
            if ( v27 - ((__int64)&v22[-*a2] >> 3) <= 1 )
            {
              v31 = v25 + 2;
              if ( v27 <= 2 * (v25 + 2) )
              {
                v38 = 1;
                if ( v27 )
                  v38 = a2[1];
                v39 = v27 + v38 + 2;
                if ( v39 > 0xFFFFFFFFFFFFFFFLL )
                  sub_4261EA(0xFFFFFFFFFFFFFFFLL, v23, v38);
                v40 = sub_22077B0(8 * v39);
                v41 = (const void *)a2[5];
                v44 = v40;
                v34 = (_QWORD *)(v40 + 8 * ((v39 - v31) >> 1));
                v42 = a2[9] + 8;
                if ( (const void *)v42 != v41 )
                  memmove(v34, v41, v42 - (_QWORD)v41);
                j_j___libc_free_0(*a2);
                a2[1] = v39;
                *a2 = v44;
              }
              else
              {
                v32 = v22 + 8;
                v33 = (v27 - v31) >> 1;
                v34 = (_QWORD *)(v26 + 8 * v33);
                v35 = v32 - v23;
                if ( v23 <= (char *)v34 )
                {
                  if ( v23 != v32 )
                    memmove((char *)v34 + v24 + 8 - v35, v23, v35);
                }
                else if ( v23 != v32 )
                {
                  memmove((void *)(v26 + 8 * v33), v23, v35);
                }
              }
              a2[5] = (unsigned __int64)v34;
              v36 = *v34;
              v22 = (char *)v34 + v24;
              a2[9] = (unsigned __int64)v34 + v24;
              a2[3] = v36;
              a2[4] = v36 + 512;
              v37 = *(_QWORD *)((char *)v34 + v24);
              a2[7] = v37;
              a2[8] = v37 + 512;
            }
            *((_QWORD *)v22 + 1) = sub_22077B0(0x200u);
            v28 = (unsigned __int8 **)a2[6];
            if ( v28 )
              *v28 = a1;
            v29 = (__int64 *)(a2[9] + 8);
            a2[9] = (unsigned __int64)v29;
            v12 = *v29;
            v30 = *v29 + 512;
            a2[7] = v12;
            a2[8] = v30;
            a2[6] = v12;
          }
          else
          {
            if ( v21 )
            {
              *v21 = a1;
              v21 = (unsigned __int8 **)a2[6];
            }
            v12 = (__int64)(v21 + 1);
            a2[6] = v12;
          }
        }
      }
    }
  }
  return v12;
}
