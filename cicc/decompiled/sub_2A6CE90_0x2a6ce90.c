// Function: sub_2A6CE90
// Address: 0x2a6ce90
//
void __fastcall sub_2A6CE90(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  unsigned int v4; // r15d
  __int64 v5; // rsi
  unsigned __int8 *v6; // rax
  unsigned __int8 *v7; // rax
  unsigned __int8 *v8; // rax
  unsigned __int8 *v9; // rax
  unsigned __int8 *v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned __int8 *v14; // [rsp+0h] [rbp-A0h]
  int v15; // [rsp+8h] [rbp-98h]
  unsigned int v16; // [rsp+Ch] [rbp-94h]
  unsigned __int8 v17[8]; // [rsp+10h] [rbp-90h] BYREF
  unsigned __int64 v18; // [rsp+18h] [rbp-88h]
  unsigned int v19; // [rsp+20h] [rbp-80h]
  unsigned __int64 v20; // [rsp+28h] [rbp-78h]
  unsigned int v21; // [rsp+30h] [rbp-70h]
  __int64 v22; // [rsp+40h] [rbp-60h] BYREF
  unsigned __int64 v23; // [rsp+48h] [rbp-58h]
  unsigned int v24; // [rsp+50h] [rbp-50h]
  unsigned __int64 v25; // [rsp+58h] [rbp-48h]
  unsigned int v26; // [rsp+60h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)(v3 + 8) == 15 && (v22 = a2, *(_BYTE *)sub_2A686D0(a1 + 136, &v22) != 6) && *(_DWORD *)(a2 + 80) == 1 )
  {
    v4 = 0;
    v14 = *(unsigned __int8 **)(a2 - 64);
    v16 = **(_DWORD **)(a2 + 72);
    v15 = *(_DWORD *)(v3 + 12);
    if ( v15 )
    {
      do
      {
        while ( v16 != v4 )
        {
          v8 = sub_2A6A1C0(a1, v14, v4);
          sub_22C05A0((__int64)v17, v8);
          sub_22C05A0((__int64)&v22, v17);
          v9 = sub_2A6A1C0(a1, (unsigned __int8 *)a2, v4++);
          sub_2A639B0(a1, v9, a2, (__int64)&v22, 0x100000000LL);
          sub_22C0090((unsigned __int8 *)&v22);
          sub_22C0090(v17);
          if ( v15 == v4 )
            return;
        }
        v5 = *(_QWORD *)(a2 - 32);
        if ( *(_BYTE *)(*(_QWORD *)(v5 + 8) + 8LL) == 15 )
        {
          v10 = sub_2A6A1C0(a1, (unsigned __int8 *)a2, v16);
          sub_2A634B0(a1, v10, a2, v11, v12, v13);
        }
        else
        {
          v6 = (unsigned __int8 *)sub_2A68BC0(a1, (unsigned __int8 *)v5);
          sub_22C05A0((__int64)v17, v6);
          sub_22C05A0((__int64)&v22, v17);
          v7 = sub_2A6A1C0(a1, (unsigned __int8 *)a2, v16);
          sub_2A639B0(a1, v7, a2, (__int64)&v22, 0x100000000LL);
          if ( (unsigned int)(unsigned __int8)v22 - 4 <= 1 )
          {
            if ( v26 > 0x40 && v25 )
              j_j___libc_free_0_0(v25);
            if ( v24 > 0x40 && v23 )
              j_j___libc_free_0_0(v23);
          }
          if ( (unsigned int)v17[0] - 4 <= 1 )
          {
            if ( v21 > 0x40 && v20 )
              j_j___libc_free_0_0(v20);
            if ( v19 > 0x40 )
            {
              if ( v18 )
                j_j___libc_free_0_0(v18);
            }
          }
        }
        ++v4;
      }
      while ( v15 != v4 );
    }
  }
  else
  {
    sub_2A6A450(a1, a2);
  }
}
