// Function: sub_F8C250
// Address: 0xf8c250
//
__int64 __fastcall sub_F8C250(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 *v6; // r12
  __int64 *v7; // r13
  _QWORD *v8; // rax
  __int64 v9; // r8
  __int64 v10; // rdx
  unsigned __int64 v11; // r9
  __int64 v12; // r15
  __int64 v13; // rax
  int v14; // r12d
  __int64 v15; // rdi
  __int64 v16; // r13
  __int64 v17; // rax
  unsigned int *v18; // rax
  unsigned int *v19; // r13
  __int64 v20; // rdx
  __int64 *v22; // rax
  unsigned int v23; // [rsp+1Ch] [rbp-E4h]
  __int64 *v24; // [rsp+20h] [rbp-E0h]
  _QWORD *v25; // [rsp+20h] [rbp-E0h]
  unsigned int *v26; // [rsp+28h] [rbp-D8h]
  char v27[32]; // [rsp+30h] [rbp-D0h] BYREF
  __int16 v28; // [rsp+50h] [rbp-B0h]
  char v29[32]; // [rsp+60h] [rbp-A0h] BYREF
  __int16 v30; // [rsp+80h] [rbp-80h]
  __int64 *v31; // [rsp+90h] [rbp-70h] BYREF
  __int64 v32; // [rsp+98h] [rbp-68h]
  _BYTE v33[96]; // [rsp+A0h] [rbp-60h] BYREF

  v4 = a1 + 520;
  v31 = (__int64 *)v33;
  v6 = *(__int64 **)(a2 + 40);
  v32 = 0x600000000LL;
  v7 = &v6[*(unsigned int *)(a2 + 48)];
  if ( v7 != v6 )
  {
    do
    {
      v8 = sub_F8C220(a1, *v6, a3);
      v10 = (unsigned int)v32;
      v11 = (unsigned int)v32 + 1LL;
      if ( v11 > HIDWORD(v32) )
      {
        v25 = v8;
        sub_C8D5F0((__int64)&v31, v33, (unsigned int)v32 + 1LL, 8u, v9, v11);
        v10 = (unsigned int)v32;
        v8 = v25;
      }
      a2 = a3;
      ++v6;
      v31[v10] = (__int64)v8;
      LODWORD(v32) = v32 + 1;
      sub_D5F1F0(v4, a3);
    }
    while ( v7 != v6 );
    v23 = v32;
    if ( (_DWORD)v32 )
    {
      v24 = v31;
      v12 = *v31;
      if ( (_DWORD)v32 == 1 )
        goto LABEL_15;
      v13 = 1;
      v14 = 1;
      while ( 1 )
      {
        while ( 1 )
        {
          v15 = *(_QWORD *)(a1 + 600);
          v28 = 257;
          v16 = v24[v13];
          a2 = 29;
          v17 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v15 + 16LL))(
                  v15,
                  29,
                  v12,
                  v16);
          if ( !v17 )
            break;
          v12 = v17;
LABEL_9:
          v13 = (unsigned int)(v14 + 1);
          v14 = v13;
          if ( v23 <= (unsigned int)v13 )
            goto LABEL_14;
        }
        v30 = 257;
        v12 = sub_B504D0(29, v12, v16, (__int64)v29, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 608) + 16LL))(
          *(_QWORD *)(a1 + 608),
          v12,
          v27,
          *(_QWORD *)(v4 + 56),
          *(_QWORD *)(v4 + 64));
        v18 = *(unsigned int **)(a1 + 520);
        a2 = (__int64)&v18[4 * *(unsigned int *)(a1 + 528)];
        v19 = v18;
        v26 = (unsigned int *)a2;
        if ( v18 == (unsigned int *)a2 )
          goto LABEL_9;
        do
        {
          v20 = *((_QWORD *)v19 + 1);
          a2 = *v19;
          v19 += 4;
          sub_B99FD0(v12, a2, v20);
        }
        while ( v26 != v19 );
        v13 = (unsigned int)(v14 + 1);
        v14 = v13;
        if ( v23 <= (unsigned int)v13 )
        {
LABEL_14:
          v24 = v31;
          goto LABEL_15;
        }
      }
    }
  }
  v22 = (__int64 *)sub_BD5C60(a3);
  v12 = sub_ACD720(v22);
  v24 = v31;
LABEL_15:
  if ( v24 != (__int64 *)v33 )
    _libc_free(v24, a2);
  return v12;
}
