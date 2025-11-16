// Function: sub_18E8FE0
// Address: 0x18e8fe0
//
void __fastcall sub_18E8FE0(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 *v4; // rbx
  _QWORD *v5; // rdx
  __int64 v6; // r8
  int v7; // r9d
  __int64 *v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rax
  char *v12; // rcx
  int v13; // esi
  unsigned int v14; // eax
  __int64 v15; // rsi
  __int64 *v16; // r8
  unsigned int v17; // eax
  __int64 **v18; // rdi
  _BYTE *v19; // r12
  int v20; // edx
  _BYTE *v21; // rbx
  __int64 v22; // [rsp+8h] [rbp-398h]
  char **v24; // [rsp+18h] [rbp-388h]
  int v25; // [rsp+28h] [rbp-378h]
  __int64 v26; // [rsp+28h] [rbp-378h]
  __int64 *v27; // [rsp+28h] [rbp-378h]
  _QWORD *v28; // [rsp+28h] [rbp-378h]
  __int64 *v29; // [rsp+38h] [rbp-368h] BYREF
  char *v30; // [rsp+40h] [rbp-360h] BYREF
  unsigned int v31; // [rsp+48h] [rbp-358h]
  char *v32; // [rsp+50h] [rbp-350h] BYREF
  __int64 v33; // [rsp+58h] [rbp-348h]
  _BYTE v34[128]; // [rsp+60h] [rbp-340h] BYREF
  __int64 v35; // [rsp+E0h] [rbp-2C0h]
  __int64 *v36; // [rsp+F0h] [rbp-2B0h]
  _BYTE *v37; // [rsp+F8h] [rbp-2A8h] BYREF
  __int64 v38; // [rsp+100h] [rbp-2A0h]
  _BYTE v39[664]; // [rsp+108h] [rbp-298h] BYREF

  v4 = a2;
  v29 = a2;
  if ( (unsigned int)sub_18E8390((__int64 *)a1, a2, a3, (__int64)&v29) > 1 )
  {
    v37 = v39;
    v38 = 0x400000000LL;
    v8 = (__int64 *)v29[18];
    v36 = v8;
    v9 = *v8;
    v22 = *v8;
    if ( a3 != a2 )
    {
      while ( 1 )
      {
        v15 = v4[18];
        v16 = v8 + 3;
        LODWORD(v33) = *(_DWORD *)(v15 + 32);
        if ( (unsigned int)v33 > 0x40 )
        {
          v27 = v8 + 3;
          sub_16A4FD0((__int64)&v32, (const void **)(v15 + 24));
          v16 = v27;
        }
        else
        {
          v32 = *(char **)(v15 + 24);
        }
        sub_16A7590((__int64)&v32, v16);
        v10 = (unsigned int)v33;
        v12 = v32;
        v31 = v33;
        v30 = v32;
        v25 = v33;
        if ( (unsigned int)v33 > 0x40 )
        {
          v24 = (char **)v32;
          v10 = v25 - (unsigned int)sub_16A57B0((__int64)&v30);
          if ( (unsigned int)v10 > 0x40 )
            goto LABEL_5;
          v12 = *v24;
        }
        v11 = 0;
        if ( v12 )
LABEL_5:
          v11 = sub_15A1070(v22, (__int64)&v30);
        v13 = *((_DWORD *)v4 + 2);
        v32 = v34;
        v33 = 0x800000000LL;
        if ( v13 )
        {
          v26 = v11;
          sub_18E63F0((__int64)&v32, (char **)v4, v10, (__int64)v12, v6, v7);
          v11 = v26;
        }
        v35 = v11;
        v14 = v38;
        if ( (unsigned int)v38 >= HIDWORD(v38) )
        {
          sub_18E88A0((__int64)&v37, 0);
          v14 = v38;
        }
        v9 = 19LL * v14;
        v5 = &v37[152 * v14];
        if ( v5 )
        {
          v5[1] = 0x800000000LL;
          *v5 = v5 + 2;
          v9 = (unsigned int)v33;
          if ( (_DWORD)v33 )
          {
            v28 = v5;
            sub_18E63F0((__int64)v5, &v32, (__int64)v5, (unsigned int)v33, v6, v7);
            v5 = v28;
          }
          v5[18] = v35;
          v14 = v38;
        }
        LODWORD(v38) = v14 + 1;
        if ( v32 != v34 )
          _libc_free((unsigned __int64)v32);
        if ( v31 > 0x40 && v30 )
          j_j___libc_free_0_0(v30);
        v4 += 20;
        if ( v4 == a3 )
          break;
        v8 = v36;
      }
    }
    v17 = *(_DWORD *)(a1 + 144);
    if ( v17 >= *(_DWORD *)(a1 + 148) )
    {
      sub_18E8DE0(a1 + 136, 0);
      v17 = *(_DWORD *)(a1 + 144);
    }
    v18 = (__int64 **)(*(_QWORD *)(a1 + 136) + 632LL * v17);
    if ( v18 )
    {
      *v18 = v36;
      v18[1] = (__int64 *)(v18 + 3);
      v18[2] = (__int64 *)0x400000000LL;
      if ( !(_DWORD)v38 )
      {
        v19 = v37;
        ++*(_DWORD *)(a1 + 144);
LABEL_33:
        if ( v19 != v39 )
          _libc_free((unsigned __int64)v19);
        return;
      }
      sub_18E8A60((__int64)(v18 + 1), (__int64)&v37, (__int64)v5, v9, v6, v7);
      v20 = v38;
      v17 = *(_DWORD *)(a1 + 144);
    }
    else
    {
      v20 = v38;
    }
    *(_DWORD *)(a1 + 144) = v17 + 1;
    v21 = v37;
    v19 = &v37[152 * v20];
    if ( v37 != v19 )
    {
      do
      {
        v19 -= 152;
        if ( *(_BYTE **)v19 != v19 + 16 )
          _libc_free(*(_QWORD *)v19);
      }
      while ( v21 != v19 );
      v19 = v37;
    }
    goto LABEL_33;
  }
}
