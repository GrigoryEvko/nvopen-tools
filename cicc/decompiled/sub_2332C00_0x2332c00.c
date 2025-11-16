// Function: sub_2332C00
// Address: 0x2332c00
//
__int64 __fastcall sub_2332C00(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rdx
  __int64 v6; // rcx
  unsigned __int64 v7; // r8
  __int64 v8; // rdx
  unsigned int v9; // eax
  unsigned int v10; // ebx
  __int64 v11; // rdx
  __int64 v12; // r14
  __int64 v13; // rax
  char v15; // di
  char v16; // [rsp+Eh] [rbp-C2h]
  char v17; // [rsp+Fh] [rbp-C1h]
  __int64 v18; // [rsp+10h] [rbp-C0h] BYREF
  unsigned __int64 v19; // [rsp+18h] [rbp-B8h]
  __int64 v20; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v21; // [rsp+30h] [rbp-A0h] BYREF
  unsigned __int64 v22; // [rsp+38h] [rbp-98h]
  unsigned __int64 v23[4]; // [rsp+40h] [rbp-90h] BYREF
  _QWORD v24[4]; // [rsp+60h] [rbp-70h] BYREF
  char v25; // [rsp+80h] [rbp-50h]
  _QWORD v26[2]; // [rsp+88h] [rbp-48h] BYREF
  _QWORD *v27; // [rsp+98h] [rbp-38h] BYREF

  v18 = a2;
  v19 = a3;
  v16 = 0;
  v17 = 1;
  if ( a3 )
  {
    while ( 1 )
    {
      v21 = 0;
      v22 = 0;
      LOBYTE(v24[0]) = 59;
      v4 = sub_C931B0(&v18, v24, 1u, 0);
      if ( v4 == -1 )
      {
        v6 = v18;
        v4 = v19;
        v7 = 0;
        v8 = 0;
      }
      else
      {
        v5 = v4 + 1;
        v6 = v18;
        if ( v4 + 1 > v19 )
        {
          v5 = v19;
          v7 = 0;
        }
        else
        {
          v7 = v19 - v5;
        }
        v8 = v18 + v5;
        if ( v4 > v19 )
          v4 = v19;
      }
      v21 = v6;
      v22 = v4;
      v18 = v8;
      v19 = v7;
      if ( v4 <= 2 )
        break;
      if ( *(_WORD *)v6 == 28526 && *(_BYTE *)(v6 + 2) == 45 )
      {
        v4 -= 3LL;
        v6 += 3;
        v15 = 0;
        v21 = v6;
        v22 = v4;
      }
      else
      {
        v15 = 1;
      }
      if ( v4 == 18 )
      {
        if ( *(_QWORD *)v6 ^ 0x642D726564616568LL | *(_QWORD *)(v6 + 8) ^ 0x69746163696C7075LL
          || *(_WORD *)(v6 + 16) != 28271 )
        {
          break;
        }
        v17 = v15;
      }
      else
      {
        if ( v4 != 15
          || *(_QWORD *)v6 != 0x2D65726170657270LL
          || *(_DWORD *)(v6 + 8) != 762474342
          || *(_WORD *)(v6 + 12) != 29804
          || *(_BYTE *)(v6 + 14) != 111 )
        {
          break;
        }
        v16 = v15;
      }
      if ( !v7 )
        goto LABEL_18;
    }
    v9 = sub_C63BB0();
    v24[1] = 40;
    v10 = v9;
    v12 = v11;
    v24[0] = "invalid LoopRotate pass parameter '{0}' ";
    v24[2] = &v27;
    v24[3] = 1;
    v25 = 1;
    v26[0] = &unk_49DB108;
    v26[1] = &v21;
    v27 = v26;
    sub_23328D0((__int64)v23, (__int64)v24);
    sub_23058C0(&v20, (__int64)v23, v10, v12);
    v13 = v20;
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v13 & 0xFFFFFFFFFFFFFFFELL;
    sub_2240A30(v23);
  }
  else
  {
LABEL_18:
    *(_BYTE *)(a1 + 8) = *(_BYTE *)(a1 + 8) & 0xFC | 2;
    *(_BYTE *)a1 = v17;
    *(_BYTE *)(a1 + 1) = v16;
  }
  return a1;
}
