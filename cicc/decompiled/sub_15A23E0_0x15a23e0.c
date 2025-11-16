// Function: sub_15A23E0
// Address: 0x15a23e0
//
__int64 __fastcall sub_15A23E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rsi
  int v5; // edx
  __int64 v6; // rdx
  int v7; // r9d
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r15
  __int64 v11; // r15
  _QWORD *v13; // rax
  __int64 v14; // [rsp+8h] [rbp-A8h]
  int v15; // [rsp+14h] [rbp-9Ch]
  int v16; // [rsp+18h] [rbp-98h]
  unsigned int v17; // [rsp+20h] [rbp-90h]
  __int64 *v19; // [rsp+30h] [rbp-80h] BYREF
  __int64 v20; // [rsp+38h] [rbp-78h]
  _BYTE v21[112]; // [rsp+40h] [rbp-70h] BYREF

  v4 = 0;
  v5 = *(_DWORD *)(a1 + 20);
  v19 = (__int64 *)v21;
  v6 = v5 & 0xFFFFFFF;
  v20 = 0x800000000LL;
  if ( (unsigned int)v6 > 8uLL )
  {
    sub_16CD150(&v19, v21, (unsigned int)v6, 8);
    v4 = (unsigned int)v20;
    v6 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  }
  v7 = v6;
  if ( (_DWORD)v6 )
  {
    v17 = 0;
    v8 = (unsigned int)(v6 - 1);
    v9 = 0;
    v7 = 0;
    while ( 1 )
    {
      v10 = *(_QWORD *)(a1 + 24 * (v9 - v6));
      if ( a2 == v10 )
      {
        v17 = v9;
        v10 = a3;
        ++v7;
        if ( HIDWORD(v20) > (unsigned int)v4 )
          goto LABEL_6;
      }
      else if ( HIDWORD(v20) > (unsigned int)v4 )
      {
        goto LABEL_6;
      }
      v14 = v8;
      v15 = v7;
      sub_16CD150(&v19, v21, 0, 8);
      v4 = (unsigned int)v20;
      v8 = v14;
      v7 = v15;
LABEL_6:
      v19[v4] = v10;
      v4 = (unsigned int)(v20 + 1);
      LODWORD(v20) = v20 + 1;
      if ( v8 == v9 )
        goto LABEL_12;
      ++v9;
      v6 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    }
  }
  v17 = 0;
LABEL_12:
  v16 = v7;
  v11 = sub_159ABB0(v19, v4);
  if ( !v11 )
  {
    v13 = (_QWORD *)sub_16498A0(a1);
    v11 = sub_15A1D80(*v13 + 1616LL, v19, (unsigned int)v20, (__int64 *)a1, a2, a3, v16, v17);
  }
  if ( v19 != (__int64 *)v21 )
    _libc_free((unsigned __int64)v19);
  return v11;
}
