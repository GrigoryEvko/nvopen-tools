// Function: sub_D96EA0
// Address: 0xd96ea0
//
_QWORD *__fastcall sub_D96EA0(__int64 a1, unsigned __int16 a2, unsigned __int64 *a3, __int64 a4, __int64 a5)
{
  unsigned __int64 *v5; // r15
  unsigned __int64 v6; // r12
  unsigned __int64 *v7; // r13
  __int64 v8; // rax
  _DWORD *v9; // rdx
  __int64 v10; // r9
  unsigned __int64 v11; // r12
  __int64 v12; // rax
  _QWORD *v13; // r12
  unsigned int v15; // [rsp+4h] [rbp-DCh]
  __int64 v16; // [rsp+18h] [rbp-C8h] BYREF
  _DWORD *v17; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v18; // [rsp+28h] [rbp-B8h]
  _DWORD v19[44]; // [rsp+30h] [rbp-B0h] BYREF

  v5 = &a3[a4];
  v17 = v19;
  v19[0] = a2;
  v18 = 0x2000000001LL;
  if ( a3 != v5 )
  {
    v6 = *a3;
    v7 = a3 + 1;
    v8 = 1;
    v9 = v19;
    v10 = (unsigned int)v6;
    while ( 1 )
    {
      v9[v8] = v10;
      v11 = HIDWORD(v6);
      LODWORD(v18) = v18 + 1;
      v12 = (unsigned int)v18;
      if ( (unsigned __int64)(unsigned int)v18 + 1 > HIDWORD(v18) )
      {
        sub_C8D5F0((__int64)&v17, v19, (unsigned int)v18 + 1LL, 4u, a5, v10);
        v12 = (unsigned int)v18;
      }
      v17[v12] = v11;
      v8 = (unsigned int)(v18 + 1);
      LODWORD(v18) = v18 + 1;
      if ( v5 == v7 )
        break;
      v6 = *v7;
      v10 = (unsigned int)*v7;
      if ( v8 + 1 > (unsigned __int64)HIDWORD(v18) )
      {
        v15 = *v7;
        sub_C8D5F0((__int64)&v17, v19, v8 + 1, 4u, a5, v10);
        v8 = (unsigned int)v18;
        v10 = v15;
      }
      v9 = v17;
      ++v7;
    }
  }
  v16 = 0;
  v13 = sub_C65B40(a1 + 1032, (__int64)&v17, &v16, (__int64)off_49DEA80);
  if ( v17 != v19 )
    _libc_free(v17, &v17);
  return v13;
}
