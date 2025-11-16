// Function: sub_DE02D0
// Address: 0xde02d0
//
__int64 __fastcall sub_DE02D0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  unsigned __int64 v7; // rsi
  __int64 v8; // rdx
  _QWORD *v9; // rdi
  __int64 result; // rax
  __int64 v11; // rdx
  bool v12; // zf
  __int64 v13; // rbx
  _QWORD *v14; // rax
  char v15; // dl
  unsigned __int64 v16; // rax
  __int64 v17; // rbx
  int v18; // eax
  __int64 v19; // rdx
  int v20; // r15d
  unsigned int v21; // r13d
  __int64 *v22; // r12
  __int64 v23; // r14
  __int64 v24; // r15
  __int64 v25; // r14
  __int64 v26; // r13
  unsigned int v27; // ebx
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  char v30; // al
  char v31; // r14
  unsigned __int64 v32; // rax
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // [rsp+0h] [rbp-B0h]
  unsigned __int64 v36; // [rsp+10h] [rbp-A0h]
  __int64 *v37; // [rsp+18h] [rbp-98h]
  __int64 v38; // [rsp+20h] [rbp-90h]
  __int64 v39; // [rsp+20h] [rbp-90h]
  __int64 *v40; // [rsp+20h] [rbp-90h]
  _QWORD *v42; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v43; // [rsp+48h] [rbp-68h]
  unsigned int v44; // [rsp+4Ch] [rbp-64h]
  _QWORD v45[12]; // [rsp+50h] [rbp-60h] BYREF

  v7 = (unsigned __int64)&v42;
  v8 = *(_QWORD *)(a3 + 80);
  v9 = v45;
  result = v8 - 24;
  v42 = v45;
  if ( !v8 )
    result = 0;
  v44 = 6;
  v43 = 1;
  v45[0] = result;
  LODWORD(result) = 1;
  do
  {
    v11 = (unsigned int)result;
    v12 = *(_BYTE *)(a2 + 28) == 0;
    v13 = v9[(unsigned int)result - 1];
    v43 = result - 1;
    if ( v12 )
      goto LABEL_15;
    v14 = *(_QWORD **)(a2 + 8);
    a4 = *(unsigned int *)(a2 + 20);
    v11 = (__int64)&v14[a4];
    if ( v14 != (_QWORD *)v11 )
    {
      while ( v13 != *v14 )
      {
        if ( (_QWORD *)v11 == ++v14 )
          goto LABEL_30;
      }
LABEL_9:
      result = v43;
LABEL_10:
      v9 = v42;
      continue;
    }
LABEL_30:
    if ( (unsigned int)a4 >= *(_DWORD *)(a2 + 16) )
    {
LABEL_15:
      v7 = v13;
      sub_C8CC70(a2, v13, v11, a4, a5, a6);
      if ( !v15 )
        goto LABEL_9;
    }
    else
    {
      *(_DWORD *)(a2 + 20) = a4 + 1;
      *(_QWORD *)v11 = v13;
      ++*(_QWORD *)a2;
    }
    a5 = v13 + 48;
    v16 = *(_QWORD *)(v13 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    a4 = v16;
    if ( v13 + 48 == v16 || !v16 || (unsigned int)*(unsigned __int8 *)(v16 - 24) - 30 > 0xA )
      BUG();
    if ( *(_BYTE *)(v16 - 24) == 31 && (*(_DWORD *)(v16 - 20) & 0x7FFFFFF) == 3 )
    {
      v24 = *(_QWORD *)(v16 - 120);
      if ( v24 )
      {
        v25 = *(_QWORD *)(v16 - 56);
        if ( v25 )
        {
          v26 = *(_QWORD *)(v16 - 88);
          if ( v26 )
          {
            if ( *(_BYTE *)v24 == 17 )
            {
              v27 = *(_DWORD *)(v24 + 32);
              if ( v27 <= 0x40 )
              {
                v28 = v43;
                a4 = v44;
                v29 = v43 + 1LL;
                if ( *(_QWORD *)(v24 + 24) != 1 )
                  v25 = v26;
                if ( v29 <= v44 )
                  goto LABEL_40;
              }
              else
              {
                if ( (unsigned int)sub_C444A0(v24 + 24) != v27 - 1 )
                  v25 = v26;
LABEL_39:
                v28 = v43;
                a4 = v44;
                v29 = v43 + 1LL;
                if ( v29 <= v44 )
                  goto LABEL_40;
              }
              v7 = (unsigned __int64)v45;
              sub_C8D5F0((__int64)&v42, v45, v29, 8u, a5, a6);
              v28 = v43;
LABEL_40:
              v42[v28] = v25;
              result = ++v43;
              goto LABEL_10;
            }
            if ( *(_BYTE *)v24 == 82 )
            {
              v37 = sub_DD8400((__int64)a1, *(_QWORD *)(v24 - 64));
              v40 = sub_DD8400((__int64)a1, *(_QWORD *)(v24 - 32));
              v7 = ((unsigned __int64)((*(_BYTE *)(v24 + 1) & 2) != 0) << 32)
                 | *(_WORD *)(v24 + 2) & 0x3F
                 | v36 & 0xFFFFFF0000000000LL;
              v36 = v7;
              v30 = sub_DCCA40(a1, v7, (__int64)v37, (__int64)v40);
              a5 = v13 + 48;
              if ( v30 )
                goto LABEL_39;
              v31 = *(_BYTE *)(v24 + 1) >> 1;
              v7 = ((unsigned __int64)(v31 & 1) << 32)
                 | (unsigned int)sub_B52870(*(_WORD *)(v24 + 2) & 0x3F)
                 | v35 & 0xFFFFFF0000000000LL;
              v35 = v7;
              if ( (unsigned __int8)sub_DCCA40(a1, v7, (__int64)v37, (__int64)v40) )
              {
                v33 = v43;
                a4 = v44;
                v34 = v43 + 1LL;
                if ( v34 > v44 )
                {
                  v7 = (unsigned __int64)v45;
                  sub_C8D5F0((__int64)&v42, v45, v34, 8u, a5, a6);
                  v33 = v43;
                }
                v42[v33] = v26;
                result = ++v43;
                goto LABEL_10;
              }
              v32 = *(_QWORD *)(v13 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              a4 = v32;
              if ( v13 + 48 == v32 )
                goto LABEL_41;
              if ( !v32 )
                BUG();
            }
          }
        }
      }
    }
    v17 = a4 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(a4 - 24) - 30 <= 0xA )
    {
      v18 = sub_B46E30(a4 - 24);
      v19 = v43;
      a5 = v18;
      a6 = v43 + (__int64)v18;
      v20 = v18;
      if ( a6 > v44 )
        goto LABEL_42;
      goto LABEL_23;
    }
LABEL_41:
    v19 = v43;
    a5 = 0;
    v20 = 0;
    v17 = 0;
    a6 = v43;
    if ( v43 > (unsigned __int64)v44 )
    {
LABEL_42:
      v7 = (unsigned __int64)v45;
      v39 = a5;
      sub_C8D5F0((__int64)&v42, v45, a6, 8u, a5, a6);
      v19 = v43;
      a5 = v39;
    }
LABEL_23:
    v9 = v42;
    if ( v20 )
    {
      v38 = a2;
      v21 = 0;
      v22 = &v42[v19];
      v23 = a5;
      do
      {
        if ( v22 )
        {
          v7 = v21;
          *v22 = sub_B46EC0(v17, v21);
        }
        ++v21;
        ++v22;
      }
      while ( v21 != v20 );
      a2 = v38;
      LODWORD(v19) = v43;
      a5 = v23;
      v9 = v42;
    }
    v43 = a5 + v19;
    result = (unsigned int)(a5 + v19);
  }
  while ( (_DWORD)result );
  if ( v9 != v45 )
    return _libc_free(v9, v7);
  return result;
}
