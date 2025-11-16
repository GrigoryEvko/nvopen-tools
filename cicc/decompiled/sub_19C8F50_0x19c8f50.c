// Function: sub_19C8F50
// Address: 0x19c8f50
//
void __fastcall sub_19C8F50(
        __int64 a1,
        char *a2,
        unsigned int a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // r10
  __int64 v17; // r15
  __int64 v18; // r9
  size_t v19; // rdx
  __int64 *v20; // r15
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rbx
  int v24; // r8d
  int v25; // r9d
  __int64 v26; // rax
  __int64 *v27; // rax
  __int64 *v28; // rax
  unsigned __int8 *v29; // r13
  double v30; // xmm4_8
  double v31; // xmm5_8
  __int64 v32; // [rsp+0h] [rbp-90h]
  __int64 v33; // [rsp+8h] [rbp-88h]
  __int64 v34; // [rsp+10h] [rbp-80h]
  size_t v35; // [rsp+18h] [rbp-78h]
  __int64 v36[2]; // [rsp+20h] [rbp-70h] BYREF
  __int64 *v37; // [rsp+30h] [rbp-60h] BYREF
  __int64 v38; // [rsp+38h] [rbp-58h]
  _QWORD v39[10]; // [rsp+40h] [rbp-50h] BYREF

  v37 = v39;
  v38 = 0x400000001LL;
  v39[0] = 0;
  v12 = sub_13FD000(a1);
  if ( v12 )
  {
    v13 = *(unsigned int *)(v12 + 8);
    v14 = v12;
    if ( (unsigned int)v13 > 1 )
    {
      v15 = (unsigned int)v38;
      v16 = v13;
      v17 = 1;
      while ( 1 )
      {
        v18 = *(_QWORD *)(v14 + 8 * (v17 - v13));
        if ( (unsigned int)v15 >= HIDWORD(v38) )
        {
          v34 = v14;
          v32 = v16;
          v33 = *(_QWORD *)(v14 + 8 * (v17 - v13));
          sub_16CD150((__int64)&v37, v39, 0, 8, v14, v18);
          v15 = (unsigned int)v38;
          v16 = v32;
          v18 = v33;
          v14 = v34;
        }
        ++v17;
        v37[v15] = v18;
        v15 = (unsigned int)(v38 + 1);
        LODWORD(v38) = v38 + 1;
        if ( v16 == v17 )
          break;
        v13 = *(unsigned int *)(v14 + 8);
      }
    }
  }
  v19 = 0;
  if ( a2 )
    v19 = strlen(a2);
  v35 = v19;
  v20 = (__int64 *)sub_157E9C0(**(_QWORD **)(a1 + 32));
  v36[0] = sub_161FF10(v20, a2, v35);
  v21 = sub_1643350(v20);
  v22 = sub_159C470(v21, a3, 0);
  v36[1] = (__int64)sub_1624210(v22);
  v23 = sub_1627350(v20, v36, (__int64 *)2, 0, 1);
  v26 = (unsigned int)v38;
  if ( (unsigned int)v38 >= HIDWORD(v38) )
  {
    sub_16CD150((__int64)&v37, v39, 0, 8, v24, v25);
    v26 = (unsigned int)v38;
  }
  v37[v26] = v23;
  v27 = *(__int64 **)(a1 + 32);
  LODWORD(v38) = v38 + 1;
  v28 = (__int64 *)sub_157E9C0(*v27);
  v29 = (unsigned __int8 *)sub_1627350(v28, v37, (__int64 *)(unsigned int)v38, 0, 1);
  sub_1630830((__int64)v29, 0, v29, a4, a5, a6, a7, v30, v31, a10, a11);
  sub_13FCC30(a1, (__int64)v29);
  if ( v37 != v39 )
    _libc_free((unsigned __int64)v37);
}
