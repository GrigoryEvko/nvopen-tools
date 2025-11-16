// Function: sub_15A66D0
// Address: 0x15a66d0
//
__int64 __fastcall sub_15A66D0(__int64 a1, __int64 *a2, int a3)
{
  __int64 v3; // rax
  __int64 *v4; // rbx
  unsigned int v5; // ecx
  __int64 v6; // rdx
  __int64 v7; // r15
  _BYTE *v8; // r12
  __int64 v9; // rcx
  _BYTE *v10; // rsi
  __int64 v11; // r12
  _BYTE *v13; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v14; // [rsp+18h] [rbp-B8h]
  _BYTE v15[176]; // [rsp+20h] [rbp-B0h] BYREF

  v13 = v15;
  v14 = 0x1000000000LL;
  if ( a3 )
  {
    v3 = (unsigned int)(a3 - 1);
    v4 = a2;
    v5 = 16;
    v6 = 0;
    v7 = (__int64)&a2[v3 + 1];
    while ( 1 )
    {
      v8 = (_BYTE *)*v4;
      if ( *v4 && (unsigned __int8)(*v8 - 4) <= 0x1Eu )
      {
        if ( (unsigned int)v6 >= v5 )
        {
          sub_16CD150(&v13, v15, 0, 8);
          v6 = (unsigned int)v14;
        }
        ++v4;
        *(_QWORD *)&v13[8 * v6] = v8;
        v6 = (unsigned int)(v14 + 1);
        LODWORD(v14) = v14 + 1;
        if ( v4 == (__int64 *)v7 )
        {
LABEL_12:
          v10 = v13;
          goto LABEL_13;
        }
      }
      else
      {
        if ( (unsigned int)v6 >= v5 )
        {
          sub_16CD150(&v13, v15, 0, 8);
          v6 = (unsigned int)v14;
        }
        v9 = *v4++;
        *(_QWORD *)&v13[8 * v6] = v9;
        v6 = (unsigned int)(v14 + 1);
        LODWORD(v14) = v14 + 1;
        if ( v4 == (__int64 *)v7 )
          goto LABEL_12;
      }
      v5 = HIDWORD(v14);
    }
  }
  v6 = 0;
  v10 = v15;
LABEL_13:
  v11 = sub_1627350(*(_QWORD *)(a1 + 8), v10, v6, 0, 1);
  if ( v13 != v15 )
    _libc_free((unsigned __int64)v13);
  return v11;
}
