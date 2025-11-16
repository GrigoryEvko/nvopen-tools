// Function: sub_155FDB0
// Address: 0x155fdb0
//
__int64 __fastcall sub_155FDB0(__int64 *a1, __int64 a2, __int64 a3, _QWORD *a4, unsigned __int64 a5)
{
  __int64 v5; // r9
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // r14
  __int64 v9; // rdx
  unsigned int v10; // ebx
  __int64 v11; // rax
  unsigned int v12; // eax
  signed __int64 v13; // r14
  __int64 v14; // rbx
  __int64 *v15; // rsi
  __int64 v16; // r12
  __int64 v19; // [rsp+18h] [rbp-88h]
  __int64 *v20; // [rsp+20h] [rbp-80h] BYREF
  __int64 v21; // [rsp+28h] [rbp-78h]
  _QWORD v22[14]; // [rsp+30h] [rbp-70h] BYREF

  v5 = a3;
  v6 = a5;
  v7 = a5;
  while ( 1 )
  {
    if ( !v6 )
      goto LABEL_5;
    if ( a4[v6 - 1] )
      break;
    --v6;
  }
  v10 = v6 + 2;
  if ( (_DWORD)v6 == -2 )
  {
LABEL_5:
    if ( !a3 )
    {
      v16 = 0;
      if ( !a2 )
        return v16;
      v9 = 1;
      v22[0] = a2;
      v20 = v22;
      v21 = 0x800000001LL;
      goto LABEL_24;
    }
    LODWORD(v9) = 1;
    v22[0] = a2;
    v10 = 2;
    v20 = v22;
    v21 = 0x800000001LL;
    goto LABEL_7;
  }
  v20 = v22;
  v21 = 0x800000000LL;
  if ( v10 <= 8 )
  {
    v22[0] = a2;
    v9 = (unsigned int)(v21 + 1);
    LODWORD(v21) = v21 + 1;
    if ( (_DWORD)v6 == -1 )
    {
LABEL_24:
      v15 = v20;
      goto LABEL_25;
    }
LABEL_7:
    if ( HIDWORD(v21) > (unsigned int)v9 )
      goto LABEL_8;
    goto LABEL_31;
  }
  sub_16CD150(&v20, v22, v10, 8);
  v5 = a3;
  if ( HIDWORD(v21) <= (unsigned int)v21 )
  {
    sub_16CD150(&v20, v22, 0, 8);
    v5 = a3;
    v20[(unsigned int)v21] = a2;
    LODWORD(v9) = v21 + 1;
    LODWORD(v21) = v21 + 1;
    goto LABEL_7;
  }
  v20[(unsigned int)v21] = a2;
  v12 = v21 + 1;
  LODWORD(v21) = v12;
  if ( v12 < HIDWORD(v21) )
  {
    v20[v12] = a3;
    v11 = (unsigned int)(v21 + 1);
    LODWORD(v21) = v21 + 1;
    goto LABEL_15;
  }
LABEL_31:
  v19 = v5;
  sub_16CD150(&v20, v22, 0, 8);
  v5 = v19;
LABEL_8:
  v20[(unsigned int)v21] = v5;
  LODWORD(v21) = v21 + 1;
  v9 = (unsigned int)v21;
  v11 = (unsigned int)v21;
  if ( v10 == 2 )
    goto LABEL_24;
LABEL_15:
  if ( v10 - 2 <= v7 )
    v7 = v10 - 2;
  v13 = 8 * v7;
  v14 = v13 >> 3;
  if ( v13 >> 3 > (unsigned __int64)HIDWORD(v21) - v11 )
  {
    sub_16CD150(&v20, v22, v11 + v14, 8);
    v11 = (unsigned int)v21;
  }
  v15 = v20;
  if ( v13 )
  {
    memcpy(&v20[v11], a4, v13);
    LODWORD(v11) = v21;
    v15 = v20;
  }
  LODWORD(v21) = v14 + v11;
  v9 = (unsigned int)(v14 + v11);
LABEL_25:
  v16 = sub_155F990(a1, v15, v9);
  if ( v20 != v22 )
    _libc_free((unsigned __int64)v20);
  return v16;
}
