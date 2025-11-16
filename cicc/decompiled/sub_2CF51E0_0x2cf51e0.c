// Function: sub_2CF51E0
// Address: 0x2cf51e0
//
__int64 __fastcall sub_2CF51E0(_QWORD *a1, __int64 a2)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 *v7; // r9
  __int64 v8; // rsi
  __int64 v9; // rax
  unsigned int v10; // r13d
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax

  sub_2CE3910((__int64)a1, a2);
  v8 = *(_QWORD *)(a2 + 80);
  if ( v8 != a2 + 72 )
  {
    while ( 1 )
    {
      if ( !v8 )
        BUG();
      v9 = *(_QWORD *)(v8 + 32);
      v5 = v8 + 24;
      if ( v9 != v8 + 24 )
        break;
LABEL_9:
      v8 = *(_QWORD *)(v8 + 8);
      if ( a2 + 72 == v8 )
        goto LABEL_10;
    }
    while ( 1 )
    {
      if ( !v9 )
        BUG();
      if ( *(_BYTE *)(v9 - 24) == 61 )
      {
        v4 = *(_QWORD *)(v9 - 16);
        if ( *(_BYTE *)(v4 + 8) == 14 )
        {
          v4 = *(_DWORD *)(v4 + 8) >> 8;
          if ( !(_DWORD)v4 )
            break;
        }
      }
      v9 = *(_QWORD *)(v9 + 8);
      if ( v5 == v9 )
        goto LABEL_9;
    }
    sub_2CF4300(a1, a2);
    if ( (_BYTE)qword_5013EE8 )
      sub_2CE51D0(a1, a2);
  }
LABEL_10:
  if ( (_DWORD)qword_5013E08 == 1 )
    v10 = sub_2CEAC10(a1, a2);
  else
    v10 = sub_2CF2C20(a1, a2, v4, v5, v6, v7);
  v11 = a1[64];
  while ( v11 )
  {
    sub_2CDE810(*(_QWORD *)(v11 + 24));
    v12 = v11;
    v11 = *(_QWORD *)(v11 + 16);
    j_j___libc_free_0(v12);
  }
  a1[64] = 0;
  a1[65] = a1 + 63;
  a1[66] = a1 + 63;
  v13 = a1[3];
  a1[67] = 0;
  if ( v13 != a1[4] )
    a1[4] = v13;
  v14 = a1[6];
  if ( v14 != a1[7] )
    a1[7] = v14;
  v15 = a1[9];
  if ( v15 != a1[10] )
    a1[10] = v15;
  v16 = a1[12];
  if ( v16 != a1[13] )
    a1[13] = v16;
  return v10;
}
