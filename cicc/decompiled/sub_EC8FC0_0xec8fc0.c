// Function: sub_EC8FC0
// Address: 0xec8fc0
//
__int64 __fastcall sub_EC8FC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdi
  _QWORD *v5; // rsi
  unsigned int v6; // r12d
  __int64 v7; // rdi
  char *v8; // rdx
  _BYTE *v9; // rax
  int v10; // ecx
  __int64 *v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi
  _BYTE *v14; // r13
  __int64 v15; // rbx
  _QWORD *v16; // rbx
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rdi
  void (*v21)(); // rax
  char *v22; // [rsp+8h] [rbp-158h]
  _QWORD v25[2]; // [rsp+20h] [rbp-140h] BYREF
  _QWORD v26[2]; // [rsp+30h] [rbp-130h] BYREF
  _QWORD v27[2]; // [rsp+40h] [rbp-120h] BYREF
  __int64 v28; // [rsp+50h] [rbp-110h]
  __int64 v29; // [rsp+58h] [rbp-108h]
  __int16 v30; // [rsp+60h] [rbp-100h]
  _QWORD v31[2]; // [rsp+70h] [rbp-F0h] BYREF
  char *v32; // [rsp+80h] [rbp-E0h]
  __int16 v33; // [rsp+90h] [rbp-D0h]
  _BYTE *v34; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v35; // [rsp+A8h] [rbp-B8h]
  _BYTE v36[176]; // [rsp+B0h] [rbp-B0h] BYREF

  v34 = v36;
  v35 = 0x400000000LL;
  while ( 1 )
  {
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 3 )
    {
      v18 = *(_QWORD *)(a1 + 8);
      v30 = 1283;
      v27[0] = "expected string in '";
      v33 = 770;
      v5 = v31;
      v28 = a2;
      v29 = a3;
      v31[0] = v27;
      v32 = "' directive";
      v6 = sub_ECE0E0(v18, v31, 0, 0);
      goto LABEL_18;
    }
    v4 = *(_QWORD *)(a1 + 8);
    v5 = v25;
    v25[1] = 0;
    v25[0] = v26;
    LOBYTE(v26[0]) = 0;
    v6 = (*(__int64 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v4 + 208LL))(v4, v25);
    if ( (_BYTE)v6 )
      goto LABEL_16;
    v7 = (unsigned int)v35;
    v8 = (char *)v25;
    v9 = v34;
    v5 = (_QWORD *)((unsigned int)v35 + 1LL);
    v10 = v35;
    if ( (unsigned __int64)v5 > HIDWORD(v35) )
    {
      if ( v34 > (_BYTE *)v25 || v25 >= (_QWORD *)&v34[32 * (unsigned int)v35] )
      {
        sub_95D880((__int64)&v34, (__int64)v5);
        v7 = (unsigned int)v35;
        v9 = v34;
        v8 = (char *)v25;
        v10 = v35;
      }
      else
      {
        v22 = (char *)((char *)v25 - v34);
        sub_95D880((__int64)&v34, (__int64)v5);
        v9 = v34;
        v7 = (unsigned int)v35;
        v8 = &v22[(_QWORD)v34];
        v10 = v35;
      }
    }
    v11 = (__int64 *)&v9[32 * v7];
    if ( v11 )
    {
      *v11 = (__int64)(v11 + 2);
      v5 = *(_QWORD **)v8;
      sub_EC5090(v11, *(_BYTE **)v8, *(_QWORD *)v8 + *((_QWORD *)v8 + 1));
      v10 = v35;
    }
    v12 = *(_QWORD *)(a1 + 8);
    LODWORD(v35) = v10 + 1;
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(__int64, _QWORD *, char *))(*(_QWORD *)v12 + 40LL))(v12, v5, v8) + 8) == 9 )
      break;
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 26 )
    {
      v13 = *(_QWORD *)(a1 + 8);
      v27[0] = "unexpected token in '";
      v5 = v31;
      v33 = 770;
      v28 = a2;
      v29 = a3;
      v30 = 1283;
      v31[0] = v27;
      v32 = "' directive";
      v6 = sub_ECE0E0(v13, v31, 0, 0);
LABEL_16:
      if ( (_QWORD *)v25[0] != v26 )
      {
        v5 = (_QWORD *)(v26[0] + 1LL);
        j_j___libc_free_0(v25[0], v26[0] + 1LL);
      }
LABEL_18:
      v14 = v34;
      v15 = (unsigned int)v35;
      goto LABEL_19;
    }
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    if ( (_QWORD *)v25[0] != v26 )
      j_j___libc_free_0(v25[0], v26[0] + 1LL);
  }
  if ( (_QWORD *)v25[0] != v26 )
  {
    v5 = (_QWORD *)(v26[0] + 1LL);
    j_j___libc_free_0(v25[0], v26[0] + 1LL);
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  v14 = v34;
  v15 = (unsigned int)v35;
  v20 = v19;
  v21 = *(void (**)())(*(_QWORD *)v19 + 232LL);
  if ( v21 != nullsub_100 )
  {
    v5 = v34;
    ((void (__fastcall *)(__int64, _BYTE *, _QWORD))v21)(v20, v34, (unsigned int)v35);
    v14 = v34;
    v15 = (unsigned int)v35;
  }
LABEL_19:
  v16 = &v14[32 * v15];
  if ( v16 != (_QWORD *)v14 )
  {
    do
    {
      v16 -= 4;
      if ( (_QWORD *)*v16 != v16 + 2 )
      {
        v5 = (_QWORD *)(v16[2] + 1LL);
        j_j___libc_free_0(*v16, v5);
      }
    }
    while ( v16 != (_QWORD *)v14 );
    v14 = v34;
  }
  if ( v14 != v36 )
    _libc_free(v14, v5);
  return v6;
}
