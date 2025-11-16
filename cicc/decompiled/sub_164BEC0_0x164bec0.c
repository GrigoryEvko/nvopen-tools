// Function: sub_164BEC0
// Address: 0x164bec0
//
__int64 __fastcall sub_164BEC0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  unsigned __int64 v13; // rdi
  __int64 v15; // r13
  __int64 v16; // r14
  __int64 v17; // r15
  __int64 v18; // rbx
  double v19; // xmm4_8
  double v20; // xmm5_8
  double v21; // xmm4_8
  double v22; // xmm5_8
  __int64 v23; // rbx
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // rsi
  __int64 v27; // rdi
  __int64 v28; // r13
  __int64 v29; // rax
  __int64 v30; // r15
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r14
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // [rsp+8h] [rbp-48h]
  __int64 v37; // [rsp+10h] [rbp-40h]
  __int64 v38; // [rsp+18h] [rbp-38h]

  switch ( *(_BYTE *)(a1 + 16) )
  {
    case 0:
      sub_15E3C20((_QWORD *)a1);
      return sub_1648B90(a1);
    case 1:
    case 2:
      sub_159D9E0(a1);
      sub_164BE60(a1, a5, a6, a7, a8, v19, v20, a11, a12);
      return sub_1648B90(a1);
    case 3:
      sub_15E5530(a1);
      sub_159D9E0(a1);
      sub_164BE60(a1, a5, a6, a7, a8, v21, v22, a11, a12);
      *(_DWORD *)(a1 + 20) = *(_DWORD *)(a1 + 20) & 0xF0000000 | 1;
      return sub_1648B90(a1);
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 0xA:
    case 0xF:
    case 0x10:
      goto LABEL_14;
    case 0xB:
    case 0xC:
      v15 = *(_QWORD *)(a1 + 32);
      if ( v15 )
      {
        v16 = *(_QWORD *)(v15 + 32);
        if ( v16 )
        {
          v17 = *(_QWORD *)(v16 + 32);
          if ( v17 )
          {
            v18 = *(_QWORD *)(v17 + 32);
            if ( v18 )
            {
              sub_16042C0(*(_QWORD *)(v17 + 32));
              sub_1648B90(v18);
            }
            sub_164BE60(v17, a5, a6, a7, a8, a9, a10, a11, a12);
            sub_1648B90(v17);
          }
          sub_164BE60(v16, a5, a6, a7, a8, a9, a10, a11, a12);
          sub_1648B90(v16);
        }
        sub_164BE60(v15, a5, a6, a7, a8, a9, a10, a11, a12);
        sub_1648B90(v15);
      }
      goto LABEL_14;
    case 0xD:
      if ( *(_DWORD *)(a1 + 32) > 0x40u )
      {
        v27 = *(_QWORD *)(a1 + 24);
        if ( v27 )
          j_j___libc_free_0_0(v27);
      }
      goto LABEL_14;
    case 0xE:
      v23 = sub_16982C0(a1, a2, a3, a4);
      if ( *(_QWORD *)(a1 + 32) == v23 )
      {
        v28 = *(_QWORD *)(a1 + 40);
        if ( v28 )
        {
          v29 = 32LL * *(_QWORD *)(v28 - 8);
          v30 = v28 + v29;
          while ( v28 != v30 )
          {
            v30 -= 32;
            if ( v23 == *(_QWORD *)(v30 + 8) )
            {
              v31 = *(_QWORD *)(v30 + 16);
              v37 = v31;
              if ( v31 )
              {
                v32 = 32LL * *(_QWORD *)(v31 - 8);
                v33 = v31 + v32;
                if ( v31 != v31 + v32 )
                {
                  do
                  {
                    v33 -= 32;
                    if ( v23 == *(_QWORD *)(v33 + 8) )
                    {
                      v34 = *(_QWORD *)(v33 + 16);
                      v36 = v34;
                      if ( v34 )
                      {
                        v35 = v34 + 32LL * *(_QWORD *)(v34 - 8);
                        if ( v34 != v35 )
                        {
                          do
                          {
                            v38 = v35 - 32;
                            sub_127D120((_QWORD *)(v35 - 24));
                            v35 = v38;
                          }
                          while ( v36 != v38 );
                        }
                        j_j_j___libc_free_0_0(v36 - 8);
                      }
                    }
                    else
                    {
                      sub_1698460(v33 + 8);
                    }
                  }
                  while ( v37 != v33 );
                }
                j_j_j___libc_free_0_0(v37 - 8);
              }
            }
            else
            {
              sub_1698460(v30 + 8);
            }
          }
          j_j_j___libc_free_0_0(v28 - 8);
        }
      }
      else
      {
        sub_1698460(a1 + 32);
      }
LABEL_14:
      sub_164BE60(a1, a5, a6, a7, a8, a9, a10, a11, a12);
      return sub_1648B90(a1);
    case 0x11:
      sub_164BE60(a1, a5, a6, a7, a8, a9, a10, a11, a12);
      v26 = 40;
      return j_j___libc_free_0(a1, v26);
    case 0x12:
      sub_157EF40(a1);
      v26 = 64;
      return j_j___libc_free_0(a1, v26);
    case 0x13:
      sub_161E830(a1);
      v26 = 32;
      return j_j___libc_free_0(a1, v26);
    case 0x14:
      v24 = *(_QWORD *)(a1 + 56);
      if ( v24 != a1 + 72 )
        j_j___libc_free_0(v24, *(_QWORD *)(a1 + 72) + 1LL);
      v25 = *(_QWORD *)(a1 + 24);
      if ( v25 != a1 + 40 )
        j_j___libc_free_0(v25, *(_QWORD *)(a1 + 40) + 1LL);
      sub_164BE60(a1, a5, a6, a7, a8, a9, a10, a11, a12);
      v26 = 104;
      return j_j___libc_free_0(a1, v26);
    case 0x15:
    case 0x16:
    case 0x17:
      return (*(__int64 (**)(void))(a1 + 24))();
    case 0x18:
    case 0x19:
    case 0x1A:
    case 0x1B:
    case 0x1C:
    case 0x1D:
    case 0x1E:
    case 0x1F:
    case 0x20:
    case 0x21:
    case 0x22:
    case 0x23:
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x27:
    case 0x28:
    case 0x29:
    case 0x2A:
    case 0x2B:
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x2F:
    case 0x30:
    case 0x31:
    case 0x32:
    case 0x33:
    case 0x34:
    case 0x35:
    case 0x36:
    case 0x37:
    case 0x38:
    case 0x39:
    case 0x3A:
    case 0x3B:
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x3F:
    case 0x40:
    case 0x41:
    case 0x42:
    case 0x43:
    case 0x44:
    case 0x45:
    case 0x46:
    case 0x47:
    case 0x48:
    case 0x49:
    case 0x4A:
    case 0x4B:
    case 0x4C:
    case 0x4D:
    case 0x4E:
    case 0x4F:
    case 0x50:
    case 0x51:
    case 0x52:
    case 0x53:
    case 0x54:
    case 0x55:
    case 0x58:
      goto LABEL_4;
    case 0x56:
    case 0x57:
      v13 = *(_QWORD *)(a1 + 56);
      if ( v13 != a1 + 72 )
        _libc_free(v13);
LABEL_4:
      sub_15F2000(a1);
      return sub_1648B90(a1);
  }
}
