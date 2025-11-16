// Function: sub_17D7560
// Address: 0x17d7560
//
char __fastcall sub_17D7560(__int128 a1)
{
  __int64 v1; // r14
  __int64 v2; // rbx
  __int64 *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  _QWORD *v6; // r15
  _BYTE *v7; // r12
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 *v11; // r9
  int v12; // eax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 *v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 *v20; // [rsp+0h] [rbp-60h]
  __int64 v21; // [rsp+8h] [rbp-58h]
  __int64 *v22; // [rsp+8h] [rbp-58h]
  _QWORD v23[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v24; // [rsp+20h] [rbp-40h]

  v1 = 0;
  v2 = a1;
  *(_QWORD *)&a1 = *(_QWORD *)(a1 + 24);
  v3 = sub_17D4DA0(a1);
  v6 = *(_QWORD **)(v2 + 24);
  v7 = v3;
  if ( *(_DWORD *)(v6[1] + 156LL) )
  {
    v18 = sub_17D4880(*(_QWORD *)(v2 + 24), *((const char **)&a1 + 1), v4, v5);
    v6 = *(_QWORD **)(v2 + 24);
    v1 = v18;
  }
  if ( *(_QWORD *)v2 )
  {
    v8 = sub_17CF940(v6, *(__int64 **)(v2 + 16), v7, **(_QWORD **)v2, 0);
    *(_QWORD *)&a1 = *(_QWORD *)(v2 + 16);
    *((_QWORD *)&a1 + 1) = *(_QWORD *)v2;
    v7 = (_BYTE *)v8;
    v23[0] = "_msprop";
    v24 = 259;
    v9 = sub_156D390((__int64 *)a1, *((__int64 *)&a1 + 1), v8, (__int64)v23);
    v6 = *(_QWORD **)(v2 + 24);
    *(_QWORD *)v2 = v9;
  }
  else
  {
    *(_QWORD *)v2 = v7;
  }
  v10 = v6[1];
  if ( *(_DWORD *)(v10 + 156) )
  {
    if ( *(_QWORD *)(v2 + 8) )
    {
      if ( *(_BYTE *)(v1 + 16) <= 0x10u )
      {
        LOBYTE(v10) = sub_1593BB0(v1, *((__int64 *)&a1 + 1), v4, v5);
        if ( (_BYTE)v10 )
          return v10;
        v6 = *(_QWORD **)(v2 + 24);
      }
      v11 = *(__int64 **)(v2 + 16);
      if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) == 16 )
      {
        v20 = *(__int64 **)(v2 + 16);
        v21 = *(_QWORD *)v7;
        v12 = sub_1643030(*(_QWORD *)(*(_QWORD *)v7 + 24LL));
        v13 = sub_1644900(*(_QWORD **)(v6[1] + 168LL), *(_DWORD *)(v21 + 32) * v12);
        if ( v21 != v13 )
        {
          v24 = 257;
          v7 = (_BYTE *)sub_12AA3B0(v20, 0x2Fu, (__int64)v7, v13, (__int64)v23);
        }
        v11 = *(__int64 **)(v2 + 16);
        v6 = *(_QWORD **)(v2 + 24);
      }
      v22 = v11;
      v24 = 257;
      v14 = sub_17CDAE0(v6, *(_QWORD *)v7);
      v15 = sub_12AA0C0(v22, 0x21u, v7, v14, (__int64)v23);
      v16 = *(__int64 **)(v2 + 16);
      v17 = *(_QWORD *)(v2 + 8);
      v24 = 257;
      v10 = sub_156B790(v16, v15, v1, v17, (__int64)v23, 0);
      *(_QWORD *)(v2 + 8) = v10;
    }
    else
    {
      *(_QWORD *)(v2 + 8) = v1;
    }
  }
  return v10;
}
