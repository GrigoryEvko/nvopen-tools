// Function: sub_2A6A920
// Address: 0x2a6a920
//
void __fastcall sub_2A6A920(_BYTE **a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 v3; // r8
  __int64 v4; // r9
  unsigned __int64 v5; // rax
  __int64 v6; // rbx
  unsigned __int8 *v7; // rax
  __int64 v8; // rax
  __int64 v9; // r9
  __int64 v10; // rdx
  unsigned __int64 v11; // r8
  __int64 *v12; // rdi
  __int64 v13; // rax
  int v14; // ebx
  __int64 v15; // rax
  __int64 *v16; // r13
  unsigned __int8 *v17; // rax
  unsigned __int8 *v18; // rdx
  __int64 v19; // [rsp+8h] [rbp-C8h]
  __int64 v20; // [rsp+10h] [rbp-C0h]
  unsigned __int8 v21[48]; // [rsp+20h] [rbp-B0h] BYREF
  __int64 *v22; // [rsp+50h] [rbp-80h] BYREF
  __int64 v23; // [rsp+58h] [rbp-78h]
  _BYTE v24[112]; // [rsp+60h] [rbp-70h] BYREF

  v22 = (__int64 *)a2;
  if ( *(_BYTE *)sub_2A686D0((__int64)(a1 + 17), (__int64 *)&v22) == 6 )
    goto LABEL_18;
  v2 = sub_2A68BC0((__int64)a1, *(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( *(_BYTE *)v2 > 1u )
  {
    if ( *(_BYTE *)v2 == 3 && sub_AC30F0(v2[1]) )
    {
      if ( sub_B4DE50(a2) )
        goto LABEL_25;
      if ( sub_B4DE30(a2) )
      {
        v13 = *(_QWORD *)(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) + 8LL);
        if ( (unsigned int)*(unsigned __int8 *)(v13 + 8) - 17 <= 1 )
          v13 = **(_QWORD **)(v13 + 16);
        v14 = *(_DWORD *)(v13 + 8) >> 8;
        v15 = sub_B43CB0(a2);
        if ( !sub_B2F070(v15, v14) )
        {
LABEL_25:
          v22 = (__int64 *)a2;
          v16 = sub_2A686D0((__int64)(a1 + 17), (__int64 *)&v22);
          v17 = (unsigned __int8 *)sub_AD6530(*(_QWORD *)(a2 + 8), (__int64)&v22);
          sub_2A63460((__int64)a1, v16, a2, v17);
          return;
        }
      }
LABEL_18:
      sub_2A6A450((__int64)a1, a2);
      return;
    }
    v22 = (__int64 *)v24;
    v23 = 0x800000000LL;
    v5 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    if ( v5 > 8 )
    {
      sub_C8D5F0((__int64)&v22, v24, v5, 8u, v3, v4);
      v5 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    }
    if ( (_DWORD)v5 )
    {
      v6 = 0;
      v20 = (unsigned int)(v5 - 1);
      while ( 1 )
      {
        v7 = (unsigned __int8 *)sub_2A68BC0((__int64)a1, *(unsigned __int8 **)(a2 + 32 * (v6 - v5)));
        sub_22C05A0((__int64)v21, v7);
        if ( v21[0] <= 1u )
          break;
        v8 = sub_2A637C0(
               (__int64)a1,
               (__int64)v21,
               *(_QWORD *)(*(_QWORD *)(a2 + 32 * (v6 - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) + 8LL));
        if ( !v8 )
        {
          sub_2A6A450((__int64)a1, a2);
          break;
        }
        v10 = (unsigned int)v23;
        v11 = (unsigned int)v23 + 1LL;
        if ( v11 > HIDWORD(v23) )
        {
          v19 = v8;
          sub_C8D5F0((__int64)&v22, v24, (unsigned int)v23 + 1LL, 8u, v11, v9);
          v10 = (unsigned int)v23;
          v8 = v19;
        }
        v22[v10] = v8;
        LODWORD(v23) = v23 + 1;
        sub_22C0090(v21);
        if ( v6 == v20 )
          goto LABEL_26;
        ++v6;
        v5 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
      }
      sub_22C0090(v21);
      v12 = v22;
      if ( v22 != (__int64 *)v24 )
        goto LABEL_16;
    }
    else
    {
LABEL_26:
      v18 = (unsigned __int8 *)sub_97D230((unsigned __int8 *)a2, v22, (unsigned int)v23, *a1, 0, 1u);
      if ( v18 )
        sub_2A68820((__int64)a1, a2, v18);
      else
        sub_2A6A450((__int64)a1, a2);
      v12 = v22;
      if ( v22 != (__int64 *)v24 )
LABEL_16:
        _libc_free((unsigned __int64)v12);
    }
  }
}
