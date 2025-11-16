// Function: sub_F61C40
// Address: 0xf61c40
//
__int64 __fastcall sub_F61C40(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r15
  __int64 v15; // rdx
  unsigned __int8 v16; // bl
  __int64 v18; // r15
  int v19; // r14d
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rdi
  __int64 v27; // [rsp+0h] [rbp-90h]
  unsigned __int8 v28; // [rsp+Fh] [rbp-81h]
  __int64 v29[8]; // [rsp+10h] [rbp-80h] BYREF
  __int16 v30; // [rsp+50h] [rbp-40h]

  v28 = sub_F50EE0((unsigned __int8 *)a1, a4);
  if ( v28 )
  {
    v18 = 0;
    sub_F54ED0((unsigned __int8 *)a1);
    v19 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
    if ( v19 )
    {
      do
      {
        if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
          v20 = *(_QWORD *)(a1 - 8) + 32 * v18;
        else
          v20 = a1 + 32 * (v18 - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
        v26 = *(_QWORD *)v20;
        if ( *(_QWORD *)v20 )
        {
          v21 = *(_QWORD *)(v20 + 8);
          **(_QWORD **)(v20 + 16) = v21;
          if ( v21 )
            *(_QWORD *)(v21 + 16) = *(_QWORD *)(v20 + 16);
        }
        *(_QWORD *)v20 = 0;
        if ( !*(_QWORD *)(v26 + 16) && a1 != v26 && *(_BYTE *)v26 > 0x1Cu )
        {
          v29[0] = v26;
          if ( sub_F50EE0((unsigned __int8 *)v26, a4) )
            sub_F61600(a2, v29, v22, v23, v24, v25);
        }
        ++v18;
      }
      while ( v19 != (_DWORD)v18 );
    }
    sub_B43D60((_QWORD *)a1);
  }
  else
  {
    v29[0] = a3;
    memset(&v29[1], 0, 56);
    v30 = 257;
    v27 = sub_1020E10(a1, v29, v7, v8, v9, v10);
    if ( v27 )
    {
      v14 = *(_QWORD *)(a1 + 16);
      if ( v14 )
      {
        do
        {
          v15 = *(_QWORD *)(v14 + 24);
          if ( a1 != v15 )
          {
            v29[0] = *(_QWORD *)(v14 + 24);
            sub_F61600(a2, v29, v15, v11, v12, v13);
          }
          v14 = *(_QWORD *)(v14 + 8);
        }
        while ( v14 );
        if ( *(_QWORD *)(a1 + 16) )
        {
          sub_BD84D0(a1, v27);
          v28 = 1;
        }
      }
      v16 = sub_F50EE0((unsigned __int8 *)a1, a4);
      if ( v16 )
      {
        sub_B43D60((_QWORD *)a1);
        return v16;
      }
    }
  }
  return v28;
}
