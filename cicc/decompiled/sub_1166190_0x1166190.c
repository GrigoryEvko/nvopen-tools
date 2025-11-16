// Function: sub_1166190
// Address: 0x1166190
//
unsigned __int8 *__fastcall sub_1166190(__m128i *a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // r15
  unsigned int v5; // ebx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rbx
  __int64 v15; // rcx
  __int64 v16; // rcx
  __int64 v17; // r12
  __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // rdx
  _BYTE *v21; // rdi
  _BYTE *v22; // rdi
  _BYTE *v23; // [rsp+8h] [rbp-58h]
  int v24; // [rsp+14h] [rbp-4Ch]
  unsigned __int8 *v25; // [rsp+18h] [rbp-48h]
  __int64 v26[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a2 - 32);
  v4 = *(_QWORD *)(a2 + 8);
  v23 = *(_BYTE **)(a2 - 64);
  if ( *(_BYTE *)v3 > 0x15u || *(_BYTE *)(v4 + 8) != 17 || (v24 = *(_DWORD *)(v4 + 32)) == 0 )
  {
LABEL_10:
    v11 = sub_F11DB0(a1->m128i_i64, (unsigned __int8 *)a2);
    if ( !v11 )
    {
      v11 = a2;
      v12 = sub_1165C10(*(_QWORD *)(a2 - 32), a1, a2, v9, v10);
      if ( v12 )
      {
        if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
          v13 = *(_QWORD *)(a2 - 8);
        else
          v13 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
        v14 = *(_QWORD *)(v13 + 32);
        if ( v14 )
        {
          v15 = *(_QWORD *)(v13 + 40);
          **(_QWORD **)(v13 + 48) = v15;
          if ( v15 )
            *(_QWORD *)(v15 + 16) = *(_QWORD *)(v13 + 48);
        }
        *(_QWORD *)(v13 + 32) = v12;
        v16 = *(_QWORD *)(v12 + 16);
        *(_QWORD *)(v13 + 40) = v16;
        if ( v16 )
          *(_QWORD *)(v16 + 16) = v13 + 40;
        *(_QWORD *)(v13 + 48) = v12 + 16;
        *(_QWORD *)(v12 + 16) = v13 + 32;
        if ( *(_BYTE *)v14 > 0x1Cu )
        {
          v17 = a1[2].m128i_i64[1];
          v26[0] = v14;
          v18 = v17 + 2096;
          sub_11604F0(v18, v26);
          v19 = *(_QWORD *)(v14 + 16);
          if ( v19 )
          {
            if ( !*(_QWORD *)(v19 + 8) )
            {
              v26[0] = *(_QWORD *)(v19 + 24);
              sub_11604F0(v18, v26);
            }
          }
        }
      }
      else if ( !(unsigned __int8)sub_1165540((__int64)a1, a2) )
      {
        if ( *v23 <= 0x15u && *v23 != 5 && !(unsigned __int8)sub_AD6CA0((__int64)v23) && *(_BYTE *)v3 == 86 )
        {
          v20 = (*(_BYTE *)(v3 + 7) & 0x40) != 0 ? *(_QWORD *)(v3 - 8) : v3 - 32LL * (*(_DWORD *)(v3 + 4) & 0x7FFFFFF);
          v21 = *(_BYTE **)(v20 + 32);
          if ( *v21 <= 0x15u && *v21 != 5 && !(unsigned __int8)sub_AD6CA0((__int64)v21) )
          {
            v22 = *(_BYTE **)(sub_986520(v3) + 64);
            if ( *v22 <= 0x15u && *v22 != 5 && !(unsigned __int8)sub_AD6CA0((__int64)v22) )
              return sub_F26350((__int64)a1, (_BYTE *)a2, v3, 1);
          }
        }
        return 0;
      }
    }
    return (unsigned __int8 *)v11;
  }
  v5 = 0;
  while ( 1 )
  {
    v6 = sub_AD69F0((unsigned __int8 *)v3, v5);
    if ( v6 )
    {
      v25 = (unsigned __int8 *)v6;
      if ( sub_AC30F0(v6) || (unsigned int)*v25 - 12 <= 1 )
        break;
    }
    if ( ++v5 == v24 )
      goto LABEL_10;
  }
  v7 = sub_ACADE0((__int64 **)v4);
  return sub_F162A0((__int64)a1, a2, v7);
}
