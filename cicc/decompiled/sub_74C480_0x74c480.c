// Function: sub_74C480
// Address: 0x74c480
//
void __fastcall sub_74C480(__int64 a1, __int64 a2)
{
  char v2; // al
  void (__fastcall *v3)(_QWORD); // rax
  __int64 v4; // r13
  void (__fastcall *v5)(__int64, _QWORD); // rax
  __int64 v6; // rdi

  if ( a1 )
  {
    v2 = *(_BYTE *)(a1 + 28);
    switch ( v2 )
    {
      case 6:
        v5 = *(void (__fastcall **)(__int64, _QWORD))(a2 + 40);
        v6 = *(_QWORD *)(a1 + 32);
        if ( v5 )
          v5(v6, 0);
        else
          sub_74C3E0(v6, (__int64 (__fastcall **)(char *, _QWORD))a2);
        break;
      case 16:
        v3 = *(void (__fastcall **)(_QWORD))(a2 + 48);
        v4 = *(_QWORD *)(a1 + 32);
        if ( v3 )
        {
          v3(*(_QWORD *)(a1 + 32));
        }
        else
        {
          sub_74C480(*(_QWORD *)(v4 + 40));
          sub_74C010(v4, 6, a2);
          (*(void (__fastcall **)(char *, __int64))a2)("::", a2);
        }
        break;
      case 3:
        sub_74C380(*(_QWORD *)(a1 + 32), (__int64 (__fastcall **)(char *, _QWORD))a2);
        break;
    }
  }
}
