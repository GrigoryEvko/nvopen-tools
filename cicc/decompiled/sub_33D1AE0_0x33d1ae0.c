// Function: sub_33D1AE0
// Address: 0x33d1ae0
//
__int64 __fastcall sub_33D1AE0(__int64 a1, char a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 i; // rdx
  __int64 *v7; // r15
  __int64 *v8; // rbx
  char v9; // cl
  __int64 v10; // rdx
  int v11; // eax
  __int64 v12; // rsi
  unsigned __int16 *v13; // rdx
  int v14; // eax
  __int64 v15; // r14
  unsigned int v16; // r14d
  unsigned int v17; // r13d
  unsigned int v18; // eax
  char v19; // si
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned int v27; // eax
  __int64 v29; // rdx
  unsigned int v30; // ebx
  unsigned __int8 v31; // [rsp+Fh] [rbp-81h]
  __int16 v32; // [rsp+10h] [rbp-80h] BYREF
  __int64 v33; // [rsp+18h] [rbp-78h]
  unsigned __int64 v34; // [rsp+20h] [rbp-70h] BYREF
  __int64 v35; // [rsp+28h] [rbp-68h]
  __int64 v36; // [rsp+30h] [rbp-60h]
  __int64 v37; // [rsp+38h] [rbp-58h]
  unsigned __int64 v38; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v39; // [rsp+48h] [rbp-48h]
  char v40; // [rsp+50h] [rbp-40h]

  v5 = a1;
  for ( i = *(unsigned int *)(a1 + 24); (_DWORD)i == 234; i = *(unsigned int *)(v5 + 24) )
    v5 = **(_QWORD **)(v5 + 40);
  v31 = ((_DWORD)i == 168) & (a2 ^ 1);
  if ( !v31 )
  {
    if ( (_DWORD)i != 156 )
      return v31;
    v7 = *(__int64 **)(v5 + 40);
    v8 = &v7[5 * *(unsigned int *)(v5 + 64)];
    if ( v7 == v8 )
      return v31;
    v9 = 1;
    while ( 1 )
    {
      v10 = *v7;
      v11 = *(_DWORD *)(*v7 + 24);
      if ( v11 != 51 )
        break;
LABEL_24:
      v7 += 5;
      if ( v8 == v7 )
        return (unsigned __int8)v9 ^ 1u;
    }
    if ( v11 == 35 || v11 == 11 )
    {
      v12 = *(_QWORD *)(v10 + 96);
      v39 = *(_DWORD *)(v12 + 32);
      if ( v39 > 0x40 )
      {
        v12 += 24;
        sub_C43780((__int64)&v38, (const void **)v12);
      }
      else
      {
        v38 = *(_QWORD *)(v12 + 24);
      }
    }
    else
    {
      if ( v11 != 36 && v11 != 12 )
        return v31;
      v12 = *(_QWORD *)(v10 + 96) + 24LL;
      if ( *(void **)v12 == sub_C33340() )
        sub_C3E660((__int64)&v34, v12);
      else
        sub_C3A850((__int64)&v34, (__int64 *)v12);
      v39 = v35;
      v38 = v34;
    }
    v40 = 1;
    v13 = *(unsigned __int16 **)(v5 + 48);
    v14 = *v13;
    v15 = *((_QWORD *)v13 + 1);
    v32 = v14;
    v33 = v15;
    if ( (_WORD)v14 )
    {
      if ( (unsigned __int16)(v14 - 17) > 0xD3u )
      {
        LOWORD(v34) = v14;
        v35 = v15;
LABEL_18:
        if ( (_WORD)v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
          BUG();
        v16 = v39;
        v17 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v14 - 16];
        if ( v39 > 0x40 )
        {
LABEL_21:
          v18 = sub_C44590((__int64)&v38);
          v19 = v40;
          if ( v17 > v18 )
            goto LABEL_40;
LABEL_22:
          if ( v19 )
          {
            v40 = 0;
            if ( v16 > 0x40 )
            {
              if ( v38 )
                j_j___libc_free_0_0(v38);
            }
          }
          v9 = 0;
          goto LABEL_24;
        }
LABEL_35:
        _RSI = v38;
        v27 = 64;
        __asm { tzcnt   rdi, rsi }
        v19 = v40;
        if ( v38 )
          v27 = _RDI;
        if ( v16 <= v27 )
          v27 = v16;
        if ( v17 > v27 )
        {
LABEL_40:
          if ( v19 )
          {
            v40 = 0;
            if ( v16 > 0x40 )
              goto LABEL_42;
          }
          return v31;
        }
        goto LABEL_22;
      }
      LOWORD(v14) = word_4456580[v14 - 1];
      v29 = 0;
    }
    else
    {
      if ( !sub_30070B0((__int64)&v32) )
      {
        v35 = v15;
        LOWORD(v34) = 0;
        goto LABEL_34;
      }
      LOWORD(v14) = sub_3009970((__int64)&v32, v12, v21, v22, v23);
    }
    LOWORD(v34) = v14;
    v35 = v29;
    if ( (_WORD)v14 )
      goto LABEL_18;
LABEL_34:
    v24 = sub_3007260((__int64)&v34);
    v16 = v39;
    v36 = v24;
    v37 = v25;
    v17 = v24;
    if ( v39 > 0x40 )
      goto LABEL_21;
    goto LABEL_35;
  }
  v39 = 1;
  v38 = 0;
  v31 = sub_33D1410(v5, (__int64)&v38, i, a4, a5);
  if ( v31 )
  {
    v30 = v39;
    if ( v39 <= 0x40 )
      return v38 == 0;
    v31 = v30 == (unsigned int)sub_C444A0((__int64)&v38);
  }
  else if ( v39 <= 0x40 )
  {
    return v31;
  }
LABEL_42:
  if ( v38 )
    j_j___libc_free_0_0(v38);
  return v31;
}
