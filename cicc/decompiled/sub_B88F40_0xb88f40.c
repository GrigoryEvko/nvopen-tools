// Function: sub_B88F40
// Address: 0xb88f40
//
__int64 __fastcall sub_B88F40(__int64 a1, __int64 *a2, char a3)
{
  __int64 *v6; // rax
  __int64 *v7; // rsi
  unsigned int v8; // r11d
  __int64 *v9; // r12
  __int64 *v10; // rbx
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 *v21; // r12
  __int64 *v22; // r14
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 result; // rax
  __int64 *v27; // rdi
  __int64 *v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 *v32; // [rsp+8h] [rbp-1D8h]
  unsigned int v33; // [rsp+28h] [rbp-1B8h]
  __int64 *v34; // [rsp+30h] [rbp-1B0h] BYREF
  __int64 v35; // [rsp+38h] [rbp-1A8h]
  _BYTE v36[64]; // [rsp+40h] [rbp-1A0h] BYREF
  __int64 *v37; // [rsp+80h] [rbp-160h] BYREF
  __int64 v38; // [rsp+88h] [rbp-158h]
  _BYTE v39[64]; // [rsp+90h] [rbp-150h] BYREF
  __int64 *v40; // [rsp+D0h] [rbp-110h] BYREF
  __int64 v41; // [rsp+D8h] [rbp-108h]
  _BYTE v42[96]; // [rsp+E0h] [rbp-100h] BYREF
  __int64 *v43; // [rsp+140h] [rbp-A0h] BYREF
  __int64 v44; // [rsp+148h] [rbp-98h]
  _BYTE v45[144]; // [rsp+150h] [rbp-90h] BYREF

  v6 = (__int64 *)sub_22077B0(32);
  v7 = v6;
  if ( v6 )
  {
    *v6 = 0;
    v6[1] = 0;
    v6[2] = 0;
    v6[3] = a1;
  }
  sub_BB9580(a2, v6);
  v40 = (__int64 *)v42;
  v41 = 0xC00000000LL;
  if ( a3 )
  {
    v8 = *(_DWORD *)(a1 + 384);
    v44 = 0xC00000000LL;
    v43 = (__int64 *)v45;
    v34 = (__int64 *)v36;
    v37 = (__int64 *)v39;
    v35 = 0x800000000LL;
    v38 = 0x800000000LL;
    v33 = v8;
    sub_B88D90(a1, (__int64)&v34, (__int64)&v37, a2);
    v9 = v34;
    if ( v34 != &v34[(unsigned int)v35] )
    {
      v32 = a2;
      v10 = &v34[(unsigned int)v35];
      do
      {
        while ( 1 )
        {
          v15 = *v9;
          v16 = *(_QWORD *)(*(_QWORD *)(*v9 + 8) + 24LL);
          if ( v33 != *(_DWORD *)(v16 + 384) )
            break;
          v17 = (unsigned int)v44;
          v18 = (unsigned int)v44 + 1LL;
          if ( v18 > HIDWORD(v44) )
          {
            sub_C8D5F0(&v43, v45, v18, 8);
            v17 = (unsigned int)v44;
          }
          ++v9;
          v43[v17] = v15;
          LODWORD(v44) = v44 + 1;
          if ( v10 == v9 )
            goto LABEL_16;
        }
        if ( v33 <= *(_DWORD *)(v16 + 384) )
          BUG();
        v11 = (unsigned int)v41;
        v12 = (unsigned int)v41 + 1LL;
        if ( v12 > HIDWORD(v41) )
        {
          sub_C8D5F0(&v40, v42, v12, 8);
          v11 = (unsigned int)v41;
        }
        v40[v11] = v15;
        v13 = *(unsigned int *)(a1 + 248);
        v14 = *(unsigned int *)(a1 + 252);
        LODWORD(v41) = v41 + 1;
        if ( v13 + 1 > v14 )
        {
          sub_C8D5F0(a1 + 240, a1 + 256, v13 + 1, 8);
          v13 = *(unsigned int *)(a1 + 248);
        }
        ++v9;
        *(_QWORD *)(*(_QWORD *)(a1 + 240) + 8 * v13) = v15;
        ++*(_DWORD *)(a1 + 248);
      }
      while ( v10 != v9 );
LABEL_16:
      a2 = v32;
    }
    v19 = (*(__int64 (__fastcall **)(__int64 *))(*a2 + 120))(a2);
    v20 = (unsigned int)v44;
    if ( !v19 )
    {
      v30 = (unsigned int)v44;
      v31 = (unsigned int)v44 + 1LL;
      if ( v31 > HIDWORD(v44) )
      {
        sub_C8D5F0(&v43, v45, v31, 8);
        v30 = (unsigned int)v44;
      }
      v43[v30] = (__int64)a2;
      v20 = (unsigned int)(v44 + 1);
      LODWORD(v44) = v44 + 1;
    }
    sub_B87B60(*(_QWORD *)(a1 + 8), v43, v20, (__int64)a2);
    if ( (_DWORD)v41 )
    {
      v29 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
      sub_B87B60(*(_QWORD *)(a1 + 8), v40, (unsigned int)v41, v29);
      LODWORD(v41) = 0;
    }
    v21 = &v37[(unsigned int)v38];
    v22 = v37;
    while ( v21 != v22 )
    {
      v23 = *v22++;
      v24 = sub_B85AD0(*(_QWORD *)(a1 + 8), v23);
      v25 = (*(__int64 (**)(void))(v24 + 48))();
      (*(void (__fastcall **)(__int64, __int64 *, __int64))(*(_QWORD *)a1 + 24LL))(a1, a2, v25);
    }
    sub_B887D0(a1, a2);
    v7 = a2;
    sub_B87180(a1, (__int64)a2);
    result = *(unsigned int *)(a1 + 24);
    if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
    {
      v7 = (__int64 *)(a1 + 32);
      sub_C8D5F0(a1 + 16, a1 + 32, result + 1, 8);
      result = *(unsigned int *)(a1 + 24);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8 * result) = a2;
    v27 = v37;
    ++*(_DWORD *)(a1 + 24);
    if ( v27 != (__int64 *)v39 )
      result = _libc_free(v27, v7);
    if ( v34 != (__int64 *)v36 )
      result = _libc_free(v34, v7);
    if ( v43 != (__int64 *)v45 )
      result = _libc_free(v43, v7);
    v28 = v40;
    if ( v40 != (__int64 *)v42 )
      return _libc_free(v28, v7);
  }
  else
  {
    result = *(unsigned int *)(a1 + 24);
    if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
    {
      v7 = (__int64 *)(a1 + 32);
      sub_C8D5F0(a1 + 16, a1 + 32, result + 1, 8);
      result = *(unsigned int *)(a1 + 24);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8 * result) = a2;
    v28 = v40;
    ++*(_DWORD *)(a1 + 24);
    if ( v28 != (__int64 *)v42 )
      return _libc_free(v28, v7);
  }
  return result;
}
