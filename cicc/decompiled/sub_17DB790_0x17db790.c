// Function: sub_17DB790
// Address: 0x17db790
//
unsigned __int64 __fastcall sub_17DB790(__int128 a1, unsigned int a2)
{
  __int64 v2; // r15
  _QWORD *v4; // rax
  __int64 v5; // rax
  __int64 *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r11
  __int64 **v10; // r14
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r9
  __int64 v19; // rax
  int v20; // esi
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  unsigned __int64 result; // rax
  __int64 *v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 *v28; // rax
  __int64 v29; // [rsp+8h] [rbp-E8h]
  _BYTE *v30; // [rsp+10h] [rbp-E0h]
  __int64 v31; // [rsp+10h] [rbp-E0h]
  __int64 v32; // [rsp+10h] [rbp-E0h]
  __int64 v33; // [rsp+18h] [rbp-D8h]
  _BYTE *v34; // [rsp+20h] [rbp-D0h]
  __int64 v35; // [rsp+20h] [rbp-D0h]
  char v36; // [rsp+28h] [rbp-C8h]
  __int64 v37; // [rsp+28h] [rbp-C8h]
  _QWORD v38[2]; // [rsp+30h] [rbp-C0h] BYREF
  __int16 v39; // [rsp+40h] [rbp-B0h]
  _QWORD v40[2]; // [rsp+50h] [rbp-A0h] BYREF
  __int16 v41; // [rsp+60h] [rbp-90h]
  __int64 v42[16]; // [rsp+70h] [rbp-80h] BYREF

  v2 = *((_QWORD *)&a1 + 1);
  v36 = *(_BYTE *)(**(_QWORD **)(*((_QWORD *)&a1 + 1) - 24LL * (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF))
                 + 8LL);
  sub_17CE510((__int64)v42, *((__int64 *)&a1 + 1), 0, 0, 0);
  if ( (*(_BYTE *)(*((_QWORD *)&a1 + 1) + 23LL) & 0x40) != 0 )
    v4 = *(_QWORD **)(*((_QWORD *)&a1 + 1) - 8LL);
  else
    v4 = (_QWORD *)(*((_QWORD *)&a1 + 1) - 24LL * (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF));
  *((_QWORD *)&a1 + 1) = *v4;
  v34 = sub_17D4DA0(a1);
  if ( (*(_BYTE *)(v2 + 23) & 0x40) != 0 )
    v5 = *(_QWORD *)(v2 - 8);
  else
    v5 = v2 - 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF);
  *((_QWORD *)&a1 + 1) = *(_QWORD *)(v5 + 24);
  v6 = sub_17D4DA0(a1);
  v9 = (__int64)v6;
  if ( v36 == 9 )
  {
    v33 = (__int64)v6;
    v28 = (__int64 *)sub_1644900(*(_QWORD **)(*(_QWORD *)(a1 + 8) + 168LL), a2);
    v10 = (__int64 **)sub_16463B0(v28, 0x40 / a2);
    v41 = 257;
    *((_QWORD *)&a1 + 1) = 47;
    v34 = (_BYTE *)sub_12AA3B0(v42, 0x2Fu, (__int64)v34, (__int64)v10, (__int64)v40);
    v41 = 257;
    v9 = sub_12AA3B0(v42, 0x2Fu, v33, (__int64)v10, (__int64)v40);
  }
  else
  {
    v10 = *(__int64 ***)v34;
  }
  v30 = (_BYTE *)v9;
  v41 = 257;
  v39 = 257;
  v11 = sub_15A06D0(v10, *((__int64 *)&a1 + 1), v7, v8);
  v12 = sub_12AA0C0(v42, 0x21u, v34, v11, (__int64)v38);
  v35 = sub_12AA3B0(v42, 0x26u, v12, (__int64)v10, (__int64)v40);
  v41 = 257;
  v39 = 257;
  v15 = sub_15A06D0(v10, 38, v13, v14);
  v16 = sub_12AA0C0(v42, 0x21u, v30, v15, (__int64)v38);
  v17 = sub_12AA3B0(v42, 0x26u, v16, (__int64)v10, (__int64)v40);
  v18 = v17;
  if ( v36 == 9 )
  {
    v29 = v17;
    v26 = sub_1643310(*(_QWORD **)(*(_QWORD *)(a1 + 8) + 168LL));
    v41 = 257;
    v32 = v26;
    v27 = sub_12AA3B0(v42, 0x2Fu, v35, v26, (__int64)v40);
    v41 = 257;
    v35 = v27;
    v18 = sub_12AA3B0(v42, 0x2Fu, v29, v32, (__int64)v40);
  }
  v19 = *(_QWORD *)(v2 - 24);
  if ( *(_BYTE *)(v19 + 16) )
    BUG();
  DWORD2(a1) = *(_DWORD *)(v19 + 36);
  if ( DWORD2(a1) != 7178 )
  {
    if ( DWORD2(a1) <= 0x1C0A )
    {
      if ( DWORD2(a1) == 6428 )
      {
        DWORD2(a1) = 6426;
      }
      else if ( DWORD2(a1) <= 0x191C )
      {
        if ( DWORD2(a1) != 6426 )
          DWORD2(a1) = 6427;
      }
      else
      {
        DWORD2(a1) = 6427;
      }
    }
    else if ( DWORD2(a1) > 0x1CB7 )
    {
      DWORD2(a1) = 7349;
    }
    else if ( DWORD2(a1) > 0x1CB5 )
    {
      DWORD2(a1) = 7350;
    }
    else
    {
      v20 = -(DWORD2(a1) < 0x1C0D);
      LOBYTE(v20) = v20 & 0x56;
      DWORD2(a1) = v20 + 7349;
    }
  }
  v31 = v18;
  v21 = sub_15E26F0(*(__int64 **)(*(_QWORD *)a1 + 40LL), SDWORD2(a1), 0, 0);
  v41 = 259;
  v40[0] = "_msprop_vector_pack";
  v38[1] = v31;
  v38[0] = v35;
  v22 = sub_1285290(v42, *(_QWORD *)(v21 + 24), v21, (int)v38, 2, (__int64)v40, 0);
  v23 = v22;
  if ( v36 == 9 )
  {
    *((_QWORD *)&a1 + 1) = *(_QWORD *)v2;
    v37 = v22;
    v41 = 257;
    v25 = sub_17CD8D0((_QWORD *)a1, *((__int64 *)&a1 + 1));
    v23 = sub_12AA3B0(v42, 0x2Fu, v37, (__int64)v25, (__int64)v40);
  }
  sub_17D4920(a1, (__int64 *)v2, v23);
  result = *(unsigned int *)(*(_QWORD *)(a1 + 8) + 156LL);
  if ( (_DWORD)result )
    result = sub_17D9C10((_QWORD *)a1, v2);
  if ( v42[0] )
    return sub_161E7C0((__int64)v42, v42[0]);
  return result;
}
