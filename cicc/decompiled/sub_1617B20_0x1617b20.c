// Function: sub_1617B20
// Address: 0x1617b20
//
void __fastcall sub_1617B20(__int64 a1, __int64 a2, char a3)
{
  __int64 v5; // rbx
  _QWORD *v6; // rax
  int v7; // r11d
  unsigned __int64 v8; // r12
  _BYTE *v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 *v16; // r12
  __int64 *v17; // r14
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 *v22; // rdi
  __int64 *v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  int v27; // [rsp+28h] [rbp-1B8h]
  _BYTE *v28; // [rsp+30h] [rbp-1B0h] BYREF
  __int64 v29; // [rsp+38h] [rbp-1A8h]
  _BYTE v30[64]; // [rsp+40h] [rbp-1A0h] BYREF
  __int64 *v31; // [rsp+80h] [rbp-160h] BYREF
  __int64 v32; // [rsp+88h] [rbp-158h]
  _BYTE v33[64]; // [rsp+90h] [rbp-150h] BYREF
  __int64 *v34; // [rsp+D0h] [rbp-110h] BYREF
  __int64 v35; // [rsp+D8h] [rbp-108h]
  _BYTE v36[96]; // [rsp+E0h] [rbp-100h] BYREF
  __int64 *v37; // [rsp+140h] [rbp-A0h] BYREF
  __int64 v38; // [rsp+148h] [rbp-98h]
  _BYTE v39[144]; // [rsp+150h] [rbp-90h] BYREF

  v5 = a2;
  v6 = (_QWORD *)sub_22077B0(32);
  if ( v6 )
  {
    *v6 = 0;
    v6[1] = 0;
    v6[2] = 0;
    v6[3] = a1;
  }
  sub_1636870(a2, v6);
  v34 = (__int64 *)v36;
  v35 = 0xC00000000LL;
  if ( a3 )
  {
    v7 = *(_DWORD *)(a1 + 400);
    v38 = 0xC00000000LL;
    v37 = (__int64 *)v39;
    v28 = v30;
    v31 = (__int64 *)v33;
    v29 = 0x800000000LL;
    v32 = 0x800000000LL;
    v27 = v7;
    sub_1614A20(a1, (__int64)&v28, (__int64)&v31, a2);
    v8 = (unsigned __int64)v28;
    if ( v28 != &v28[8 * (unsigned int)v29] )
    {
      v9 = &v28[8 * (unsigned int)v29];
      do
      {
        while ( 1 )
        {
          v12 = *(_QWORD *)v8;
          if ( v27 != *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)v8 + 8LL) + 24LL) + 400LL) )
            break;
          v13 = (unsigned int)v38;
          if ( (unsigned int)v38 >= HIDWORD(v38) )
          {
            sub_16CD150(&v37, v39, 0, 8);
            v13 = (unsigned int)v38;
          }
          v8 += 8LL;
          v37[v13] = v12;
          LODWORD(v38) = v38 + 1;
          if ( v9 == (_BYTE *)v8 )
            goto LABEL_15;
        }
        v10 = (unsigned int)v35;
        if ( (unsigned int)v35 >= HIDWORD(v35) )
        {
          sub_16CD150(&v34, v36, 0, 8);
          v10 = (unsigned int)v35;
        }
        v34[v10] = v12;
        v11 = *(unsigned int *)(a1 + 264);
        LODWORD(v35) = v35 + 1;
        if ( (unsigned int)v11 >= *(_DWORD *)(a1 + 268) )
        {
          sub_16CD150(a1 + 256, a1 + 272, 0, 8);
          v11 = *(unsigned int *)(a1 + 264);
        }
        v8 += 8LL;
        *(_QWORD *)(*(_QWORD *)(a1 + 256) + 8 * v11) = v12;
        ++*(_DWORD *)(a1 + 264);
      }
      while ( v9 != (_BYTE *)v8 );
LABEL_15:
      v5 = a2;
    }
    v14 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 120LL))(v5);
    v15 = (unsigned int)v38;
    if ( !v14 )
    {
      v26 = (unsigned int)v38;
      if ( (unsigned int)v38 >= HIDWORD(v38) )
      {
        sub_16CD150(&v37, v39, 0, 8);
        v26 = (unsigned int)v38;
      }
      v37[v26] = v5;
      v15 = (unsigned int)(v38 + 1);
      LODWORD(v38) = v38 + 1;
    }
    sub_1613D20(*(_QWORD *)(a1 + 16), v37, v15, v5);
    if ( (_DWORD)v35 )
    {
      v24 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
      sub_1613D20(*(_QWORD *)(a1 + 16), v34, (unsigned int)v35, v24);
      LODWORD(v35) = 0;
    }
    v16 = &v31[(unsigned int)v32];
    v17 = v31;
    while ( v16 != v17 )
    {
      v18 = *v17++;
      v19 = sub_1614F20(*(_QWORD *)(a1 + 16), v18);
      v20 = (*(__int64 (**)(void))(v19 + 72))();
      (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a1 + 24LL))(a1, v5, v20);
    }
    sub_16145F0(a1, v5);
    sub_16176C0(a1, v5);
    v21 = *(unsigned int *)(a1 + 32);
    if ( (unsigned int)v21 >= *(_DWORD *)(a1 + 36) )
    {
      sub_16CD150(a1 + 24, a1 + 40, 0, 8);
      v21 = *(unsigned int *)(a1 + 32);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v21) = v5;
    v22 = v31;
    ++*(_DWORD *)(a1 + 32);
    if ( v22 != (__int64 *)v33 )
      _libc_free((unsigned __int64)v22);
    if ( v28 != v30 )
      _libc_free((unsigned __int64)v28);
    if ( v37 != (__int64 *)v39 )
      _libc_free((unsigned __int64)v37);
    v23 = v34;
    if ( v34 != (__int64 *)v36 )
      goto LABEL_30;
  }
  else
  {
    v25 = *(unsigned int *)(a1 + 32);
    if ( (unsigned int)v25 >= *(_DWORD *)(a1 + 36) )
    {
      sub_16CD150(a1 + 24, a1 + 40, 0, 8);
      v25 = *(unsigned int *)(a1 + 32);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v25) = a2;
    v23 = v34;
    ++*(_DWORD *)(a1 + 32);
    if ( v23 != (__int64 *)v36 )
LABEL_30:
      _libc_free((unsigned __int64)v23);
  }
}
