// Function: sub_3029850
// Address: 0x3029850
//
void __fastcall sub_3029850(__int64 a1, unsigned int *a2, size_t a3, unsigned int a4, char a5)
{
  __int64 v8; // rbx
  _QWORD *v9; // r8
  size_t v10; // rdx
  __int64 v11; // rax
  unsigned __int64 *v12; // r15
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  __int64 *v15; // rdi
  int v16; // eax
  _QWORD *v17; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v19[2]; // [rsp+20h] [rbp-90h] BYREF
  __int64 v20; // [rsp+30h] [rbp-80h] BYREF
  char v21; // [rsp+40h] [rbp-70h]
  void *s2; // [rsp+50h] [rbp-60h] BYREF
  __int64 v23; // [rsp+58h] [rbp-58h]
  _BYTE v24[16]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v25; // [rsp+70h] [rbp-40h]

  v8 = *(_QWORD *)(a1 + 1088);
  if ( v8 )
  {
    if ( a2 )
    {
      s2 = v24;
      sub_3020610((__int64 *)&s2, a2, (__int64)a2 + a3);
      v8 = *(_QWORD *)(a1 + 1088);
      v9 = s2;
      v10 = *(_QWORD *)(v8 + 72);
      if ( v10 == v23 )
      {
        if ( !v10 || (v17 = s2, v16 = memcmp(*(const void **)(v8 + 64), s2, v10), v9 = v17, !v16) )
        {
          if ( v9 != (_QWORD *)v24 )
          {
            j_j___libc_free_0((unsigned __int64)v9);
            v8 = *(_QWORD *)(a1 + 1088);
          }
          goto LABEL_14;
        }
      }
      if ( v9 != (_QWORD *)v24 )
        j_j___libc_free_0((unsigned __int64)v9);
    }
    else
    {
      v23 = 0;
      s2 = v24;
      v24[0] = 0;
      if ( !*(_QWORD *)(v8 + 72) )
        goto LABEL_14;
    }
  }
  v11 = sub_22077B0(0x60u);
  v8 = v11;
  if ( v11 )
    sub_3028ED0(v11, a2, a3);
  v12 = *(unsigned __int64 **)(a1 + 1088);
  *(_QWORD *)(a1 + 1088) = v8;
  if ( v12 )
  {
    v13 = v12[8];
    if ( (unsigned __int64 *)v13 != v12 + 10 )
      j_j___libc_free_0(v13);
    v14 = v12[7];
    if ( v14 )
      (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v14 + 8LL))(v14);
    sub_3020A40(v12[2]);
    j_j___libc_free_0((unsigned __int64)v12);
    v8 = *(_QWORD *)(a1 + 1088);
  }
LABEL_14:
  sub_3029260((__int64)v19, v8, a4, a5);
  if ( v21 )
  {
    v15 = *(__int64 **)(a1 + 224);
    s2 = v19;
    v25 = 260;
    sub_E99A90(v15, (__int64)&s2);
    if ( v21 )
    {
      v21 = 0;
      if ( (__int64 *)v19[0] != &v20 )
        j_j___libc_free_0(v19[0]);
    }
  }
}
