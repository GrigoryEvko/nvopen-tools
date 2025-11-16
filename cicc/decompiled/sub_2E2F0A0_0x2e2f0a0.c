// Function: sub_2E2F0A0
// Address: 0x2e2f0a0
//
__int64 __fastcall sub_2E2F0A0(__int64 a1, __int64 *a2)
{
  unsigned int v3; // eax
  unsigned int v4; // r12d
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 *v10; // rbx
  __int64 *v11; // r15
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 *v14; // rbx
  __int64 *v15; // r15
  __int64 v16; // rsi
  int v17; // eax
  __int64 *v18; // [rsp+10h] [rbp-80h] BYREF
  __int64 v19; // [rsp+18h] [rbp-78h]
  _BYTE v20[112]; // [rsp+20h] [rbp-70h] BYREF

  v3 = sub_BB92D0(a1, a2);
  if ( !(_BYTE)v3
    && (v4 = v3, (v6 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_5027190)) != 0)
    && (v7 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v6 + 104LL))(v6, &unk_5027190)) != 0
    && sub_23CF310(*(_QWORD *)(v7 + 256)) )
  {
    v10 = (__int64 *)a2[2];
    v11 = a2 + 1;
    v18 = (__int64 *)v20;
    v19 = 0x800000000LL;
    if ( a2 + 1 != v10 )
    {
      do
      {
        while ( 1 )
        {
          if ( !v10 )
            BUG();
          if ( (*((_BYTE *)v10 - 23) & 0x1C) != 0 )
            break;
          v10 = (__int64 *)v10[1];
          if ( v11 == v10 )
            goto LABEL_15;
        }
        v12 = (unsigned int)v19;
        v13 = (unsigned int)v19 + 1LL;
        if ( v13 > HIDWORD(v19) )
        {
          sub_C8D5F0((__int64)&v18, v20, v13, 8u, v8, v9);
          v12 = (unsigned int)v19;
        }
        v18[v12] = (__int64)(v10 - 7);
        LODWORD(v19) = v19 + 1;
        v10 = (__int64 *)v10[1];
      }
      while ( v11 != v10 );
LABEL_15:
      v14 = v18;
      v15 = &v18[(unsigned int)v19];
      if ( v18 != v15 )
      {
        do
        {
          v16 = *v14++;
          LOBYTE(v17) = sub_2E2EC50((__int64 **)a2, v16);
          v4 |= v17;
        }
        while ( v15 != v14 );
        v15 = v18;
      }
      if ( v15 != (__int64 *)v20 )
        _libc_free((unsigned __int64)v15);
    }
  }
  else
  {
    return 0;
  }
  return v4;
}
