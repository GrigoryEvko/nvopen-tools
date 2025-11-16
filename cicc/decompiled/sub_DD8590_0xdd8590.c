// Function: sub_DD8590
// Address: 0xdd8590
//
__int64 __fastcall sub_DD8590(__int64 *a1, __int64 a2, unsigned __int8 *a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v7; // rcx
  unsigned __int64 v8; // r9
  int v9; // eax
  __int64 v10; // rax
  __int64 *v11; // rbx
  __int64 v12; // r8
  __int64 *v13; // r15
  __int64 *v14; // rax
  __int64 v15; // rdx
  __int64 *v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 *v19; // [rsp+0h] [rbp-80h]
  unsigned __int8 v20; // [rsp+8h] [rbp-78h]
  __int64 *v21; // [rsp+10h] [rbp-70h] BYREF
  __int64 v22; // [rsp+18h] [rbp-68h]
  _BYTE v23[96]; // [rsp+20h] [rbp-60h] BYREF

  result = sub_98F650(a2, a2, a3, a4, a5);
  if ( (_BYTE)result )
  {
    v22 = 0x600000000LL;
    v9 = *(_DWORD *)(a2 + 4);
    v21 = (__int64 *)v23;
    v10 = 32LL * (v9 & 0x7FFFFFF);
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    {
      v12 = *(_QWORD *)(a2 - 8);
      v11 = (__int64 *)(v12 + v10);
    }
    else
    {
      v11 = (__int64 *)a2;
      v12 = a2 - v10;
    }
    if ( (__int64 *)v12 == v11 )
    {
      v17 = 0;
      v16 = (__int64 *)v23;
    }
    else
    {
      v13 = (__int64 *)v12;
      do
      {
        while ( !sub_D97040((__int64)a1, *(_QWORD *)(*v13 + 8)) )
        {
          v13 += 4;
          if ( v11 == v13 )
            goto LABEL_11;
        }
        v14 = sub_DD8400((__int64)a1, *v13);
        v15 = (unsigned int)v22;
        v8 = (unsigned int)v22 + 1LL;
        if ( v8 > HIDWORD(v22) )
        {
          v19 = v14;
          sub_C8D5F0((__int64)&v21, v23, (unsigned int)v22 + 1LL, 8u, v12, v8);
          v15 = (unsigned int)v22;
          v14 = v19;
        }
        v7 = (__int64)v21;
        v13 += 4;
        v21[v15] = (__int64)v14;
        LODWORD(v22) = v22 + 1;
      }
      while ( v11 != v13 );
LABEL_11:
      v16 = v21;
      v17 = (unsigned int)v22;
    }
    v18 = sub_D979D0(a1, v16, v17, v7, v12, v8);
    result = sub_D979F0((__int64)a1, v18, a2);
    if ( v21 != (__int64 *)v23 )
    {
      v20 = result;
      _libc_free(v21, v18);
      return v20;
    }
  }
  return result;
}
