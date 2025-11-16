// Function: sub_DB58E0
// Address: 0xdb58e0
//
__int64 __fastcall sub_DB58E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  unsigned __int64 v7; // r14
  unsigned int v8; // eax
  unsigned int v9; // ebx
  __int64 v10; // r13
  __int64 *v11; // rdx
  __int64 *v12; // r15
  unsigned int v13; // r13d
  __int64 v14; // rsi
  unsigned __int64 v15; // rax
  unsigned int v17; // ecx
  __int64 v18; // rax
  __int64 *v19; // r15
  __int64 *v20; // r13
  unsigned int v21; // eax
  __int64 v22; // rbx
  __int64 v23; // rsi
  unsigned int v24; // eax
  __int64 v27; // rdi
  unsigned int v28; // eax
  __int64 v29; // rax
  __int64 *v30; // r15
  __int64 *v31; // [rsp+8h] [rbp-88h]
  unsigned int v32; // [rsp+8h] [rbp-88h]
  unsigned int v33; // [rsp+8h] [rbp-88h]
  __int64 *v34; // [rsp+8h] [rbp-88h]
  __int64 v35; // [rsp+18h] [rbp-78h] BYREF
  unsigned __int64 v36; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v37; // [rsp+28h] [rbp-68h]
  __int64 v38; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v39; // [rsp+38h] [rbp-58h]
  unsigned __int64 v40; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v41; // [rsp+48h] [rbp-48h]
  __int64 v42; // [rsp+50h] [rbp-40h]
  unsigned int v43; // [rsp+58h] [rbp-38h]

  v6 = sub_D95540(a3);
  v35 = a2;
  v7 = sub_D97050(a2, v6);
  switch ( *(_WORD *)(a3 + 24) )
  {
    case 0:
      v23 = *(_QWORD *)(a3 + 32);
      v24 = *(_DWORD *)(v23 + 32);
      *(_DWORD *)(a1 + 8) = v24;
      if ( v24 > 0x40 )
        sub_C43780(a1, (const void **)(v23 + 24));
      else
        *(_QWORD *)a1 = *(_QWORD *)(v23 + 24);
      return a1;
    case 1:
    case 7:
      *(_DWORD *)(a1 + 8) = v7;
      if ( (unsigned int)v7 > 0x40 )
        sub_C43690(a1, 1, 0);
      else
        *(_QWORD *)a1 = 1;
      return a1;
    case 2:
    case 4:
      v8 = sub_DB55F0(a2, *(_QWORD *)(a3 + 32));
      *(_DWORD *)(a1 + 8) = v7;
      v9 = v8;
      if ( v7 <= v8 )
        goto LABEL_12;
      v10 = 1LL << v8;
      if ( (unsigned int)v7 <= 0x40 )
      {
        *(_QWORD *)a1 = 0;
LABEL_5:
        *(_QWORD *)a1 |= v10;
        return a1;
      }
      sub_C43690(a1, 0, 0);
      if ( *(_DWORD *)(a1 + 8) <= 0x40u )
        goto LABEL_5;
      *(_QWORD *)(*(_QWORD *)a1 + 8LL * (v9 >> 6)) |= v10;
      return a1;
    case 3:
      sub_DB4FC0((__int64)&v40, a2, *(_QWORD *)(a3 + 32));
      sub_C449B0(a1, (const void **)&v40, v7);
      if ( v41 > 0x40 && v40 )
        j_j___libc_free_0_0(v40);
      return a1;
    case 5:
    case 8:
      if ( (*(_BYTE *)(a3 + 28) & 2) != 0 )
        goto LABEL_14;
      v17 = sub_DB55F0(a2, **(_QWORD **)(a3 + 32));
      v18 = *(_QWORD *)(a3 + 32);
      v19 = (__int64 *)(v18 + 8);
      v20 = (__int64 *)(v18 + 8LL * *(_QWORD *)(a3 + 40));
      if ( v20 != (__int64 *)(v18 + 8) )
      {
        do
        {
          v32 = v17;
          v21 = sub_DB55F0(a2, *v19);
          v17 = v32;
          if ( v32 > v21 )
            v17 = v21;
          ++v19;
        }
        while ( v20 != v19 );
      }
      *(_DWORD *)(a1 + 8) = v7;
      if ( v7 <= v17 )
        goto LABEL_12;
      v22 = 1LL << v17;
      if ( (unsigned int)v7 <= 0x40 )
        goto LABEL_44;
      v33 = v17;
      sub_C43690(a1, 0, 0);
      if ( *(_DWORD *)(a1 + 8) <= 0x40u )
        goto LABEL_45;
      *(_QWORD *)(*(_QWORD *)a1 + 8LL * (v33 >> 6)) |= v22;
      return a1;
    case 6:
      v11 = *(__int64 **)(a3 + 32);
      if ( (*(_BYTE *)(a3 + 28) & 2) != 0 )
      {
        sub_DB4FC0((__int64)&v36, a2, *v11);
        v29 = *(_QWORD *)(a3 + 32);
        v30 = (__int64 *)(v29 + 8);
        v34 = (__int64 *)(v29 + 8LL * *(_QWORD *)(a3 + 40));
        if ( v34 != (__int64 *)(v29 + 8) )
        {
          do
          {
            sub_DB4FC0((__int64)&v38, a2, *v30);
            sub_C472A0((__int64)&v40, (__int64)&v36, &v38);
            if ( v37 > 0x40 && v36 )
              j_j___libc_free_0_0(v36);
            v36 = v40;
            v37 = v41;
            if ( v39 > 0x40 && v38 )
              j_j___libc_free_0_0(v38);
            ++v30;
          }
          while ( v34 != v30 );
        }
        *(_DWORD *)(a1 + 8) = v37;
        *(_QWORD *)a1 = v36;
        return a1;
      }
      v31 = &v11[*(_QWORD *)(a3 + 40)];
      if ( v31 == v11 )
      {
        v15 = 0;
        v13 = 0;
      }
      else
      {
        v12 = *(__int64 **)(a3 + 32);
        v13 = 0;
        do
        {
          v14 = *v12++;
          v13 += sub_DB55F0(a2, v14);
        }
        while ( v31 != v12 );
        v15 = v13;
      }
      *(_DWORD *)(a1 + 8) = v7;
      if ( v7 <= v15 )
        goto LABEL_12;
      v22 = 1LL << v13;
      if ( (unsigned int)v7 <= 0x40 )
        goto LABEL_44;
      goto LABEL_52;
    case 9:
    case 0xA:
    case 0xB:
    case 0xC:
    case 0xD:
LABEL_14:
      sub_DB53B0((_QWORD *)a1, &v35, a3);
      return a1;
    case 0xE:
      sub_DB4FC0(a1, a2, *(_QWORD *)(a3 + 32));
      return a1;
    case 0xF:
      sub_9AC3E0(
        (__int64)&v40,
        *(_QWORD *)(a3 - 8),
        *(_QWORD *)(a2 + 8),
        0,
        *(_QWORD *)(a2 + 32),
        0,
        *(_QWORD *)(a2 + 40),
        1);
      if ( v41 > 0x40 )
      {
        v28 = sub_C445E0((__int64)&v40);
        _RBX = v28;
        v13 = v28;
        if ( v43 <= 0x40 )
          goto LABEL_40;
        v27 = v42;
        if ( !v42 )
          goto LABEL_40;
      }
      else
      {
        _RBX = ~v40;
        if ( v40 == -1 )
        {
          _RBX = 64;
          v13 = 64;
        }
        else
        {
          __asm { tzcnt   rbx, rbx }
          v13 = _RBX;
          _RBX = (int)_RBX;
        }
        if ( v43 <= 0x40 )
          goto LABEL_42;
        v27 = v42;
        if ( !v42 )
          goto LABEL_42;
      }
      j_j___libc_free_0_0(v27);
      if ( v41 <= 0x40 )
        goto LABEL_42;
LABEL_40:
      if ( v40 )
        j_j___libc_free_0_0(v40);
LABEL_42:
      *(_DWORD *)(a1 + 8) = v7;
      if ( v7 <= _RBX )
      {
LABEL_12:
        if ( v7 > 0x40 )
          sub_C43690(a1, 0, 0);
        else
          *(_QWORD *)a1 = 0;
      }
      else
      {
        v22 = 1LL << v13;
        if ( (unsigned int)v7 <= 0x40 )
        {
LABEL_44:
          *(_QWORD *)a1 = 0;
LABEL_45:
          *(_QWORD *)a1 |= v22;
          return a1;
        }
LABEL_52:
        sub_C43690(a1, 0, 0);
        if ( *(_DWORD *)(a1 + 8) <= 0x40u )
          goto LABEL_45;
        *(_QWORD *)(*(_QWORD *)a1 + 8LL * (v13 >> 6)) |= v22;
      }
      return a1;
    default:
      BUG();
  }
}
