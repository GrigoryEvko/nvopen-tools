// Function: sub_15A4ED0
// Address: 0x15a4ed0
//
__int64 __fastcall sub_15A4ED0(__int64 a1, __int64 a2, __int64 a3, double a4, double a5, double a6)
{
  int v7; // eax
  __int64 v8; // rax
  __int64 v9; // r9
  __int64 v10; // r8
  unsigned int v11; // esi
  __int64 v12; // rdx
  unsigned int v13; // r14d
  int v14; // r15d
  __int64 v15; // rax
  _BYTE *v16; // rsi
  __int64 result; // rax
  _QWORD *v18; // rax
  __int64 v19; // [rsp+0h] [rbp-A0h]
  __int64 v20; // [rsp+8h] [rbp-98h]
  __int64 v21; // [rsp+10h] [rbp-90h]
  __int64 v23; // [rsp+18h] [rbp-88h]
  _BYTE *v24; // [rsp+20h] [rbp-80h] BYREF
  __int64 v25; // [rsp+28h] [rbp-78h]
  _BYTE v26[112]; // [rsp+30h] [rbp-70h] BYREF

  v25 = 0x800000000LL;
  v7 = *(_DWORD *)(a1 + 20);
  v24 = v26;
  v8 = v7 & 0xFFFFFFF;
  if ( (_DWORD)v8 )
  {
    v9 = (unsigned int)(v8 - 1);
    v10 = 0;
    v11 = 8;
    v12 = 0;
    v13 = 0;
    v14 = 0;
    while ( 1 )
    {
      v15 = *(_QWORD *)(a1 + 24 * (v10 - v8));
      if ( a2 == v15 )
      {
        v15 = a3;
        ++v14;
        v13 = v10;
        if ( (unsigned int)v12 < v11 )
          goto LABEL_4;
      }
      else if ( (unsigned int)v12 < v11 )
      {
        goto LABEL_4;
      }
      v19 = v10;
      v20 = v9;
      v21 = v15;
      sub_16CD150(&v24, v26, 0, 8);
      v12 = (unsigned int)v25;
      v10 = v19;
      v9 = v20;
      v15 = v21;
LABEL_4:
      *(_QWORD *)&v24[8 * v12] = v15;
      v12 = (unsigned int)(v25 + 1);
      LODWORD(v25) = v25 + 1;
      if ( v9 == v10 )
      {
        v16 = v24;
        goto LABEL_10;
      }
      v11 = HIDWORD(v25);
      ++v10;
      v8 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    }
  }
  v14 = 0;
  v13 = 0;
  v12 = 0;
  v16 = v26;
LABEL_10:
  result = sub_15A47B0(a1, (_BYTE **)v16, v12, *(__int64 ***)a1, 1, 0, a4, a5, a6);
  if ( !result )
  {
    v18 = (_QWORD *)sub_16498A0(a1);
    result = sub_15A4B20(*v18 + 1776LL, (__int64)v24, (unsigned int)v25, a1, a2, a3, v14, v13);
  }
  if ( v24 != v26 )
  {
    v23 = result;
    _libc_free((unsigned __int64)v24);
    return v23;
  }
  return result;
}
