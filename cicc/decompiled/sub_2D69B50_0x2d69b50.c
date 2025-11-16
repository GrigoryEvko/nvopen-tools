// Function: sub_2D69B50
// Address: 0x2d69b50
//
void __fastcall sub_2D69B50(__int64 a1)
{
  int v1; // ebx
  __int64 v2; // r12
  _QWORD *v3; // rbx
  __int64 v4; // rcx
  _QWORD *v5; // r12
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  bool v10; // zf
  unsigned int v11; // eax
  int v12; // r15d
  unsigned int v13; // ebx
  unsigned int v14; // eax
  unsigned int v15; // eax
  _QWORD v16[2]; // [rsp+8h] [rbp-88h] BYREF
  __int64 v17; // [rsp+18h] [rbp-78h]
  __int64 v18; // [rsp+20h] [rbp-70h]
  void *v19; // [rsp+30h] [rbp-60h]
  _QWORD v20[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v21; // [rsp+48h] [rbp-48h]
  __int64 v22; // [rsp+50h] [rbp-40h]

  v1 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v1 )
  {
    if ( !*(_DWORD *)(a1 + 20) )
      return;
    v2 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v2 <= 0x40 )
      goto LABEL_4;
    sub_2D64B90(a1);
    if ( *(_DWORD *)(a1 + 24) )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 8), (unsigned __int64)(unsigned int)v2 << 6, 8);
      *(_DWORD *)(a1 + 24) = 0;
      goto LABEL_35;
    }
    goto LABEL_40;
  }
  v11 = 4 * v1;
  v2 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v1) < 0x40 )
    v11 = 64;
  if ( (unsigned int)v2 > v11 )
  {
    v12 = 64;
    sub_2D64B90(a1);
    v13 = v1 - 1;
    if ( v13 )
    {
      _BitScanReverse(&v14, v13);
      v12 = 1 << (33 - (v14 ^ 0x1F));
      if ( v12 < 64 )
        v12 = 64;
    }
    if ( *(_DWORD *)(a1 + 24) != v12 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 8), (unsigned __int64)(unsigned int)v2 << 6, 8);
      v15 = sub_AF1560(4 * v12 / 3u + 1);
      *(_DWORD *)(a1 + 24) = v15;
      if ( !v15 )
      {
LABEL_35:
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        return;
      }
      *(_QWORD *)(a1 + 8) = sub_C7D670((unsigned __int64)v15 << 6, 8);
    }
LABEL_40:
    sub_2D69A40(a1);
    return;
  }
LABEL_4:
  v3 = *(_QWORD **)(a1 + 8);
  v16[0] = 2;
  v16[1] = 0;
  v4 = -4096;
  v5 = &v3[8 * v2];
  v17 = -4096;
  v19 = &unk_4A26638;
  v6 = -8192;
  v18 = 0;
  v20[0] = 2;
  v20[1] = 0;
  v21 = -8192;
  v22 = 0;
  if ( v3 == v5 )
    goto LABEL_23;
  while ( 1 )
  {
    v8 = v3[3];
    if ( v8 != v4 )
    {
      if ( v8 != v6 )
      {
        v7 = v3[7];
        if ( v7 == 0 || v7 == -4096 || v7 == -8192 )
        {
          v6 = v3[3];
        }
        else
        {
          sub_BD60C0(v3 + 5);
          v6 = v3[3];
          if ( v6 == v17 )
          {
LABEL_9:
            v3[4] = v18;
            v6 = v21;
            goto LABEL_10;
          }
        }
      }
      if ( v6 != 0 && v6 != -4096 && v6 != -8192 )
        sub_BD60C0(v3 + 1);
      v9 = v17;
      v10 = v17 == 0;
      v3[3] = v17;
      if ( v9 != -4096 && !v10 && v9 != -8192 )
        sub_BD6050(v3 + 1, v16[0] & 0xFFFFFFFFFFFFFFF8LL);
      goto LABEL_9;
    }
LABEL_10:
    v3 += 8;
    if ( v5 == v3 )
      break;
    v4 = v17;
  }
  v19 = &unk_49DB368;
  if ( v6 != -4096 && v6 != -8192 && v6 )
    sub_BD60C0(v20);
LABEL_23:
  *(_QWORD *)(a1 + 16) = 0;
  if ( v17 != -4096 && v17 != 0 && v17 != -8192 )
    sub_BD60C0(v16);
}
