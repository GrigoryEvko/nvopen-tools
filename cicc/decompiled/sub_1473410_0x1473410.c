// Function: sub_1473410
// Address: 0x1473410
//
__int64 __fastcall sub_1473410(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  unsigned __int8 v7; // cl
  __int64 v8; // rbx
  __int64 v9; // r14
  int v10; // eax
  __int64 *v11; // rsi
  _QWORD *v12; // rdi
  _QWORD *v13; // rsi
  __int64 v14; // r15
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r13
  __int64 v18; // r15
  __int64 v19; // r13
  unsigned __int8 v20; // dl
  char v21; // r8
  __int64 v22; // rbx
  unsigned __int64 v23; // r12
  unsigned __int64 v24; // rdi
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v29; // [rsp+20h] [rbp-2C0h]
  __int64 v30; // [rsp+28h] [rbp-2B8h]
  _QWORD *v31; // [rsp+30h] [rbp-2B0h]
  __int64 v32; // [rsp+40h] [rbp-2A0h]
  char v33; // [rsp+49h] [rbp-297h]
  char v34; // [rsp+4Ah] [rbp-296h]
  unsigned __int8 v35; // [rsp+4Ch] [rbp-294h]
  _BYTE *v36; // [rsp+50h] [rbp-290h] BYREF
  __int64 v37; // [rsp+58h] [rbp-288h]
  _BYTE v38[64]; // [rsp+60h] [rbp-280h] BYREF
  __int64 v39; // [rsp+A0h] [rbp-240h] BYREF
  __int64 v40; // [rsp+A8h] [rbp-238h]
  char v41; // [rsp+B0h] [rbp-230h]
  char v42[8]; // [rsp+B8h] [rbp-228h] BYREF
  __int64 v43; // [rsp+C0h] [rbp-220h]
  unsigned __int64 v44; // [rsp+C8h] [rbp-218h]
  _BYTE *v45; // [rsp+100h] [rbp-1E0h] BYREF
  __int64 v46; // [rsp+108h] [rbp-1D8h]
  _BYTE v47[464]; // [rsp+110h] [rbp-1D0h] BYREF

  v36 = v38;
  v37 = 0x800000000LL;
  sub_13F9CA0(a3, (__int64)&v36);
  v45 = v47;
  v46 = 0x400000000LL;
  v29 = sub_13FCB50(a3);
  if ( (_DWORD)v37 )
  {
    v7 = a4;
    v31 = (_QWORD *)a3;
    v8 = 0;
    v32 = 8LL * (unsigned int)v37;
    v35 = v7;
    v9 = 0;
    v33 = 0;
    v30 = 0;
    v34 = 1;
    do
    {
      v17 = *(_QWORD *)&v36[v8];
      sub_1473290((__int64)&v39, a2, v31, v17, v35);
      v18 = v39;
      if ( v18 == sub_1456E90(a2) )
      {
        v34 = 0;
      }
      else
      {
        v10 = v46;
        if ( (unsigned int)v46 >= HIDWORD(v46) )
        {
          sub_145EFE0((unsigned __int64 *)&v45, 0);
          v10 = v46;
        }
        v11 = (__int64 *)&v45[104 * v10];
        if ( v11 )
        {
          *v11 = v17;
          v12 = v11 + 4;
          v13 = v11 + 9;
          *(v13 - 8) = v39;
          *(v13 - 7) = v40;
          *((_BYTE *)v13 - 48) = v41;
          sub_16CCCB0(v12, v13, v42);
          v10 = v46;
        }
        LODWORD(v46) = v10 + 1;
      }
      v14 = v40;
      if ( v14 != sub_1456E90(a2) && v29 && (unsigned __int8)sub_15CC8F0(*(_QWORD *)(a2 + 56), v17, v29, v15, v16) )
      {
        if ( v30 )
        {
          v30 = sub_1481DB0(a2, v30, v40);
        }
        else
        {
          v30 = v40;
          v33 = v41;
        }
      }
      else if ( v9 != sub_1456E90(a2) )
      {
        if ( !v9 || (v19 = v40, v19 == sub_1456E90(a2)) )
          v9 = v40;
        else
          v9 = sub_1481A30(a2, v9, v40);
      }
      if ( v44 != v43 )
        _libc_free(v44);
      v8 += 8;
    }
    while ( v32 != v8 );
    v20 = v34;
    if ( !v30 )
    {
      if ( v9 )
      {
        v30 = v9;
      }
      else
      {
        v27 = sub_1456E90(a2);
        v20 = v34;
        v30 = v27;
      }
    }
    v21 = 0;
    if ( v33 )
      v21 = (_DWORD)v37 == 1;
    sub_146E8C0(a1, (__int64 *)&v45, v20, v30, v21);
  }
  else
  {
    v26 = sub_1456E90(a2);
    sub_146E8C0(a1, (__int64 *)&v45, 1u, v26, 0);
  }
  v22 = (__int64)v45;
  v23 = (unsigned __int64)&v45[104 * (unsigned int)v46];
  if ( v45 != (_BYTE *)v23 )
  {
    do
    {
      v23 -= 104LL;
      v24 = *(_QWORD *)(v23 + 48);
      if ( v24 != *(_QWORD *)(v23 + 40) )
        _libc_free(v24);
    }
    while ( v22 != v23 );
    v23 = (unsigned __int64)v45;
  }
  if ( (_BYTE *)v23 != v47 )
    _libc_free(v23);
  if ( v36 != v38 )
    _libc_free((unsigned __int64)v36);
  return a1;
}
