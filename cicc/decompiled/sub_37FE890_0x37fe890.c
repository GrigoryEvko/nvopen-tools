// Function: sub_37FE890
// Address: 0x37fe890
//
void __fastcall sub_37FE890(__int64 *a1, unsigned __int64 a2, int a3, __int64 a4, __m128i a5, __int64 a6, __int64 a7)
{
  _QWORD *v8; // rdi
  _BYTE *v9; // rdi
  _BYTE *v10; // r14
  unsigned int v11; // ebx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // r9
  _BYTE *v15; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v16; // [rsp+20h] [rbp-90h] BYREF
  __int64 v17; // [rsp+28h] [rbp-88h]
  unsigned __int64 v18; // [rsp+30h] [rbp-80h] BYREF
  __int64 v19; // [rsp+38h] [rbp-78h]
  _BYTE *v20; // [rsp+40h] [rbp-70h] BYREF
  __int64 v21; // [rsp+48h] [rbp-68h]
  _BYTE v22[96]; // [rsp+50h] [rbp-60h] BYREF

  v8 = (_QWORD *)a1[1];
  v20 = v22;
  v21 = 0x300000000LL;
  sub_34164A0(v8, a3, a2, (__int64)&v20, a4, a7);
  v9 = v20;
  v15 = &v20[16 * (unsigned int)v21];
  if ( v15 != v20 )
  {
    v10 = v20;
    v11 = 0;
    do
    {
      LODWORD(v17) = 0;
      LODWORD(v19) = 0;
      v12 = *((_QWORD *)v10 + 1);
      v10 += 16;
      v16 = 0;
      v18 = 0;
      sub_375AEA0(a1, *((_QWORD *)v10 - 2), v12, (__int64)&v16, (__int64)&v18, a5);
      v13 = v11++;
      sub_37604D0((__int64)a1, a2, v13, v16, v17, v14, v18, v19);
    }
    while ( v15 != v10 );
    v9 = v20;
  }
  if ( v9 != v22 )
    _libc_free((unsigned __int64)v9);
}
