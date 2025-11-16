// Function: sub_1481E10
// Address: 0x1481e10
//
__int64 __fastcall sub_1481E10(__int64 *a1, __int64 a2, _QWORD *a3, __int64 a4, __m128i a5, __m128i a6)
{
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // r9
  __int64 v13; // rax
  _BYTE *i; // rdx
  __int64 v15; // rdi
  __int64 v16; // r12
  __int64 v17; // [rsp+0h] [rbp-60h]
  _BYTE *v18; // [rsp+10h] [rbp-50h] BYREF
  __int64 v19; // [rsp+18h] [rbp-48h]
  _BYTE v20[64]; // [rsp+20h] [rbp-40h] BYREF

  if ( (a1[5] & 4) == 0 || !*((_DWORD *)a1 + 2) || !sub_13FCB50(a2) )
    return sub_1456E90((__int64)a3);
  v9 = *a1;
  v19 = 0x200000000LL;
  v10 = *((unsigned int *)a1 + 2);
  v18 = v20;
  v11 = v9 + 24 * v10;
  if ( v9 != v11 )
  {
    v12 = *(_QWORD *)(v9 + 8);
    v13 = 0;
    for ( i = v20; ; i = v18 )
    {
      *(_QWORD *)&i[8 * v13] = v12;
      LODWORD(v19) = v19 + 1;
      if ( a4 )
      {
        v15 = *(_QWORD *)(v9 + 16);
        if ( v15 )
        {
          if ( !sub_1452CB0(v15) )
            sub_146E690(a4, *(_QWORD *)(v9 + 16));
        }
      }
      v9 += 24;
      if ( v11 == v9 )
        break;
      v12 = *(_QWORD *)(v9 + 8);
      v13 = (unsigned int)v19;
      if ( (unsigned int)v19 >= HIDWORD(v19) )
      {
        v17 = *(_QWORD *)(v9 + 8);
        sub_16CD150(&v18, v20, 0, 8);
        v13 = (unsigned int)v19;
        v12 = v17;
      }
    }
  }
  v16 = sub_1481C30(a3, (__int64)&v18, a5, a6);
  if ( v18 != v20 )
    _libc_free((unsigned __int64)v18);
  return v16;
}
