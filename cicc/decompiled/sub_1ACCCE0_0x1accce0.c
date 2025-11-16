// Function: sub_1ACCCE0
// Address: 0x1accce0
//
__int64 __fastcall sub_1ACCCE0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 **v4; // rax
  __int64 v5; // rax
  unsigned int v6; // r15d
  __int64 **v7; // rax
  __int64 v8; // rax
  unsigned int v9; // r13d
  int v11; // eax
  __int64 v12; // r8
  int v13; // r13d
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rdx
  __int64 v18; // rax
  unsigned int v19; // eax
  __int64 v20; // [rsp+8h] [rbp-58h]
  __int64 v21; // [rsp+8h] [rbp-58h]
  __int64 v22; // [rsp+8h] [rbp-58h]
  __int64 v23; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v24; // [rsp+18h] [rbp-48h]
  __int64 v25; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v26; // [rsp+28h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v4 = *(__int64 ***)(a2 - 8);
  else
    v4 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v5 = **v4;
  if ( *(_BYTE *)(v5 + 8) == 16 )
    v5 = **(_QWORD **)(v5 + 16);
  v6 = *(_DWORD *)(v5 + 8) >> 8;
  if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
    v7 = *(__int64 ***)(a3 - 8);
  else
    v7 = (__int64 **)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
  v8 = **v7;
  if ( *(_BYTE *)(v8 + 8) == 16 )
    v8 = **(_QWORD **)(v8 + 16);
  v9 = sub_1ACA9E0((__int64)a1, v6, *(_DWORD *)(v8 + 8) >> 8);
  if ( !v9 )
  {
    v20 = sub_1632FA0(*(_QWORD *)(*a1 + 40));
    v11 = sub_15A9520(v20, v6);
    v12 = v20;
    v13 = 8 * v11;
    v24 = 8 * v11;
    if ( (unsigned int)(8 * v11) > 0x40 )
    {
      sub_16A4EF0((__int64)&v23, 0, 0);
      v26 = v13;
      sub_16A4EF0((__int64)&v25, 0, 0);
      v12 = v20;
    }
    else
    {
      v23 = 0;
      v26 = 8 * v11;
      v25 = 0;
    }
    v21 = v12;
    if ( (unsigned __int8)sub_1634900(a2, v12, (__int64)&v23) && (unsigned __int8)sub_1634900(a3, v21, (__int64)&v25) )
    {
      v9 = sub_1ACAA10((__int64)a1, (__int64)&v23, (__int64)&v25);
    }
    else
    {
      v14 = sub_16348C0(a3);
      v15 = sub_16348C0(a2);
      v9 = sub_1ACB220(a1, v15, v14);
      if ( !v9 )
      {
        v9 = sub_1ACA9E0((__int64)a1, *(_DWORD *)(a2 + 20) & 0xFFFFFFF, *(_DWORD *)(a3 + 20) & 0xFFFFFFF);
        if ( !v9 && (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 0 )
        {
          v16 = 0;
          v22 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
          do
          {
            if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
            {
              v17 = *(_QWORD *)(*(_QWORD *)(a3 - 8) + v16);
              if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
                goto LABEL_30;
            }
            else
            {
              v17 = *(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF) + v16);
              if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
              {
LABEL_30:
                v18 = *(_QWORD *)(a2 - 8);
                goto LABEL_31;
              }
            }
            v18 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
LABEL_31:
            v19 = sub_1ACCBA0((__int64)a1, *(_QWORD *)(v18 + v16), v17);
            if ( v19 )
            {
              v9 = v19;
              break;
            }
            v16 += 24;
          }
          while ( v16 != v22 );
        }
      }
    }
    if ( v26 > 0x40 && v25 )
      j_j___libc_free_0_0(v25);
    if ( v24 > 0x40 && v23 )
      j_j___libc_free_0_0(v23);
  }
  return v9;
}
