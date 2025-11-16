// Function: sub_F6F180
// Address: 0xf6f180
//
__int64 __fastcall sub_F6F180(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  int v7; // edx
  unsigned int v8; // eax
  unsigned int v10; // ebx
  __int64 v11; // r15
  _QWORD **v12; // rdx
  int v13; // ecx
  __int64 *v14; // rax
  __int64 v15; // rsi
  unsigned int *v16; // rbx
  __int64 v17; // rdx
  unsigned int v18; // esi
  __int64 v19; // [rsp+0h] [rbp-B0h]
  __int64 v20; // [rsp+18h] [rbp-98h]
  _QWORD v21[4]; // [rsp+20h] [rbp-90h] BYREF
  char v22; // [rsp+40h] [rbp-70h]
  char v23; // [rsp+41h] [rbp-6Fh]
  unsigned int v24[8]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v25; // [rsp+70h] [rbp-40h]

  v6 = *(_QWORD *)(a3 + 8);
  v7 = *(unsigned __int8 *)(v6 + 8);
  if ( (unsigned int)(v7 - 17) <= 1 )
    LOBYTE(v7) = *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL);
  if ( (_BYTE)v7 == 12 || (unsigned int)(a2 - 14) <= 1 )
  {
    v8 = sub_F6F040(a2);
    *(_QWORD *)v24 = "rdx.minmax";
    BYTE4(v20) = 0;
    v21[0] = a3;
    v21[1] = a4;
    v25 = 259;
    return sub_B35180(a1, v6, v8, (__int64)v21, 2u, v20, (__int64)v24);
  }
  else
  {
    v23 = 1;
    v10 = sub_F6F100(a2);
    v22 = 3;
    v21[0] = "rdx.minmax.cmp";
    if ( v10 <= 0xF )
    {
      v24[1] = 0;
      v11 = sub_B35C90(a1, v10, a3, a4, (__int64)v21, 0, v24[0], 0);
    }
    else
    {
      v11 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(a1 + 80) + 56LL))(
              *(_QWORD *)(a1 + 80),
              v10,
              a3,
              a4);
      if ( !v11 )
      {
        v25 = 257;
        v11 = (__int64)sub_BD2C40(72, unk_3F10FD0);
        if ( v11 )
        {
          v12 = *(_QWORD ***)(a3 + 8);
          v13 = *((unsigned __int8 *)v12 + 8);
          if ( (unsigned int)(v13 - 17) > 1 )
          {
            v15 = sub_BCB2A0(*v12);
          }
          else
          {
            BYTE4(v20) = (_BYTE)v13 == 18;
            LODWORD(v20) = *((_DWORD *)v12 + 8);
            v14 = (__int64 *)sub_BCB2A0(*v12);
            v15 = sub_BCE1B0(v14, v20);
          }
          sub_B523C0(v11, v15, 53, v10, a3, a4, (__int64)v24, 0, 0, 0);
        }
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
          *(_QWORD *)(a1 + 88),
          v11,
          v21,
          *(_QWORD *)(a1 + 56),
          *(_QWORD *)(a1 + 64));
        v16 = *(unsigned int **)a1;
        v19 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
        if ( *(_QWORD *)a1 != v19 )
        {
          do
          {
            v17 = *((_QWORD *)v16 + 1);
            v18 = *v16;
            v16 += 4;
            sub_B99FD0(v11, v18, v17);
          }
          while ( (unsigned int *)v19 != v16 );
        }
      }
    }
    *(_QWORD *)v24 = "rdx.minmax.select";
    v25 = 259;
    return sub_B36550((unsigned int **)a1, v11, a3, a4, (__int64)v24, 0);
  }
}
